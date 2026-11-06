"""Prepare complete pinned author-code dependencies without implicit training.

This module is only imported for the setup command. The pure plan performs no
filesystem mutation, dependency installation, model loading, or network request.
"""
from .registry import inspect_method, methods, resource
from .upstream import source_root

# Existing setup dependencies are preserved, including private author reference
# loaders. ProtoMIL's actual SAE and KMP's official CONCH code were omitted by
# the previous primary-repository-only command.
SOURCE_DEPENDENCIES = {
    "rmdl": ("keras_contrib_legacy",),
    "lpgmil": ("lpgmil_mamba111",),
    "akdpmp": ("smooth_topk",),
    "conmil": ("conch_focus",),
    "convl": ("conch_focus",),
    "libramil": ("conch_focus",),
    "muse": ("conch_focus",),
    "gmmamba": ("mambamil",),
    "sparseconvmil": ("sparseconvnet",),
    "lrmil": ("conch_focus",),
    "ahdmil": ("hdmil",),
    "augdiff": ("augdiff_ldm",),
    "focus": ("conch_focus",),
    "gatmamba": ("gatmamba_ssm", "lifelines_metric"),
    "triagemil": ("lifelines_metric",),
    "deepgraphsurv": ("lifelines_metric",),
    "c2mil": ("lifelines_metric",),
    "protosurv": ("lifelines_metric",),
    "micromil": ("dgl081",),
    "hmil": ("hmil_prior",),
    "vilamil": ("openai_clip",),
    "monai_mil": ("monai_tutorials",),
    "stablemil": ("timm_stablemil",),
    "mfemil": ("timm_mfemil",),
    "mammil": ("timm_mfemil",),
    "protomil": ("protomil_dtn",),
    "kmpmil": ("conch_kmp",),
}

PUBLIC_ASSET_ACTIONS = {
    "center_embeddings": {
        "id": "original_center_embeddings_initialization",
        "description": "Acquire the original public initialization using the existing checksum verifier.",
    },
    "c2c": {
        "id": "original_c2c_initialization",
        "description": "Acquire the original public initialization using the existing checksum verifier.",
    },
}


def source_closure(source, lock, dependencies=None):
    """Return deterministic primary-first code preparation order; reject cycles."""
    dependencies = SOURCE_DEPENDENCIES if dependencies is None else dependencies
    result, visited, active = [], set(), set()

    def visit(current):
        if current in active:
            raise ValueError("Pinned source dependency cycle: " + current)
        if current in visited:
            return
        if current not in lock:
            raise ValueError("No pinned author source for dependency: " + current)
        commit = lock[current].get("commit", "")
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            raise ValueError("Author source requires a full fixed Git commit: " + current)
        if not lock[current].get("url"):
            raise ValueError("Author source has no origin URL: " + current)
        active.add(current)
        result.append(current)
        for dependency in dependencies.get(current, ()):
            visit(dependency)
        active.remove(current)
        visited.add(current)

    if source is not None:
        visit(source)
    return result


def plan_setup(name):
    """Resolve the actual method/profile, all code pins and existing asset actions."""
    spec = inspect_method(name)
    family = spec.get("family", spec["id"])
    family_record = next(record for record in methods() if record["id"] == family)
    lock = resource("sources.lock.json")
    # Profiles with bundled code retain the historical optional reference fetch
    # if that family has a lock. Mean/Max have only packaged code and no lock.
    source = spec.get("source_id") or spec.get("external_source")
    if source is None and family in lock:
        source = family
    if source is None and family not in {"mean_mil", "max_mil"}:
        raise ValueError("Method has neither bundled-only preparation nor a pinned source: " + family)
    ordered = source_closure(source, lock)
    entries = [{"id": key, "url": lock[key]["url"], "commit": lock[key]["commit"],
                "path": str(source_root(key)), "dependencies": list(SOURCE_DEPENDENCIES.get(key, ()))}
               for key in ordered]
    actions = [dict(PUBLIC_ASSET_ACTIONS[key]) for key in ordered if key in PUBLIC_ASSET_ACTIONS]
    return {"requested_method": name, "method_id": spec["id"], "family": family,
            "source": source, "path": str(source_root(source)) if source else None,
            "sources": entries, "public_asset_actions": actions, "packaged_code_only": not entries,
            "method_status": family_record["status"], "profile_available": bool(spec.get("runnable")),
            "admission_blocker": family_record.get("blocker"),
            **({key: family_record.get(key, []) for key in
                ["prerequisites", "validation_blockers", "unsupported_variants"]}
               if "prerequisites" in family_record else {}),
            "scope": "Pinned code preparation; runtime, actual data and pretrained asset requirements remain method-specific."}


def prepare_sources(name, *, plan_only=False):
    plan = plan_setup(name)
    if plan_only:
        return {**plan, "state": "plan_only", "network_or_filesystem_writes_performed": False}
    # Validate the full plan before the first fetch, so an absent dependency pin
    # cannot leave a partially prepared primary repository.
    from .execution import fetch_source

    paths = {}
    for entry in plan["sources"]:
        paths[entry["id"]] = str(fetch_source(entry["id"], source_root(entry["id"])))
    asset_results = []
    for entry in plan["sources"]:
        if entry["id"] == "center_embeddings":
            from .center_embeddings_assets import fetch
            asset_results.append({"action": PUBLIC_ASSET_ACTIONS[entry["id"]]["id"], "result": fetch()})
        elif entry["id"] == "c2c":
            from .c2c_assets import fetch
            asset_results.append({"action": PUBLIC_ASSET_ACTIONS[entry["id"]]["id"], "result": fetch()})
    return {**plan, "state": "source_preparation_completed", "prepared_paths": paths,
            "asset_results": asset_results, "training_or_gated_weight_download_performed": False}
