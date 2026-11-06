"""Local experiment matrices and explicit H-cluster submission generation."""
import itertools
import re
import shlex
import subprocess
import sys
from pathlib import Path

from .data import Bundle
from .io import write_json
from .registry import inspect_method


def matrix(data, method_names, fms, seeds, output):
    if method_names == ["all"]:
        from .registry import methods
        method_names = [m["id"] for m in methods()]
    entries = []
    for method, fm, seed in itertools.product(method_names, fms, seeds):
        spec = inspect_method(method)
        reason = None if spec["runnable"] else (spec.get("blocker") or f"{method}: select an explicit runnable execution profile")
        try:
            bundle = Bundle(data, fm=fm)
            if spec["runnable"]:
                from .registry import check_compatibility, resolve_for_bundle
                spec = resolve_for_bundle(method, bundle)
                check_compatibility(spec, bundle)
            bundle.check()
        except (ValueError, KeyError, FileNotFoundError) as e:
            reason = str(e)
        entries.append({"data": str(Path(data).resolve()), "method": spec["id"], "family": spec.get('family', spec['id']), "fm": fm, "seed": seed,
                        "state": "blocked" if reason else "ready", "reason": reason})
    write_json(output, {"schema_version": 1, "entries": entries})
    return entries


def hcluster_script(repository, data, method, output, image, project, group, gpu=1, cpu=8, memory_mib=32768):
    # Deliberately generate a reviewable command. Submission is explicit CLI --submit.
    for path in (repository, data, output):
        if not Path(path).is_absolute():
            raise ValueError("Cluster repo/data/output paths must be absolute and visible in the job")
    if gpu < 1 or cpu < 1 or memory_mib < 1024:
        raise ValueError("Invalid requested resources")
    require = inspect_method(method)
    if not require["runnable"]:
        raise ValueError(f"Method {method} has no runnable adapter")
    command = ["rjob", "submit", "--name=milplus-" + re.sub(r"[^a-z0-9-]", "-", method.lower()),
        f"--namespace={project}", f"--charged-group={group}", f"--image={image}",
        f"--gpu={gpu}", f"--cpu={cpu}", f"--memory={memory_mib}", "-P", "1", "--enable-sshd",
        "--mount=gpfs://gpfs1/lixiangyu:/mnt/lixiangyu",
        "--mount=gpfs://gpfs1/medeval-share:/mnt/medeval-share",
        "--mount=gpfs://gpfs2/medeval-share-2:/mnt/medeval-share-2",
        "--mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public",
        "--", "bash", "-lc", (f"cd {shlex.quote(repository)} && python -m milplus run --data {shlex.quote(data)} "
        f"--method {shlex.quote(method)} --output {shlex.quote(output)} --device cuda")]
    return "#!/usr/bin/env bash\nset -euo pipefail\n# H CLI memory unit verified for brainpp 2.7.12.16: MiB.\n" + shlex.join(command) + "\n", command


def trident_command(repo, python, wsi_dir, output, fm, mag=20, patch_size=256, checkpoint=None, custom_list=None):
    repo = Path(repo).resolve()
    script = repo / "run_batch_of_slides.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    cmd = [python, str(script), "--task", "all", "--wsi_dir", str(wsi_dir), "--job_dir", str(output),
           "--patch_encoder", fm, "--mag", str(mag), "--patch_size", str(patch_size), "--search_nested"]
    if checkpoint:
        cmd += ["--patch_encoder_ckpt_path", str(checkpoint)]
    if custom_list:
        cmd += ["--custom_list_of_wsis", str(custom_list)]
    return cmd


def fetch_source(source_id, destination):
    from .registry import resource
    lock = resource("sources.lock.json")
    item = lock[source_id]
    commit = item["commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Source lock must use a full commit hash")
    dest = Path(destination).resolve()
    if dest.exists():
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=dest, text=True)
        current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()
        remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=dest, text=True).strip()
        if dirty or current != commit or remote.removesuffix('.git').rstrip('/') != item["url"].removesuffix('.git').rstrip('/'):
            raise ValueError("Destination exists with a different version, origin or local changes")
        return dest
    dest.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(dest)], check=True)
    subprocess.run(["git", "remote", "add", "origin", item["url"]], cwd=dest, check=True)
    sparse = item.get('sparse_files')
    if sparse:
        subprocess.run(['git', 'sparse-checkout', 'init', '--no-cone'], cwd=dest, check=True)
        subprocess.run(['git', 'sparse-checkout', 'set', '--no-cone', *sparse], cwd=dest, check=True)
    subprocess.run(["git", "fetch", *(["--filter=blob:none"] if sparse else []), "--depth", "1", "origin", commit], cwd=dest, check=True)
    subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=dest, check=True)
    return dest


def run_matrix(plan, output, devices, config=None, epochs=None):
    """One process per device, sequential within each device. No shared RNG or model state."""
    import concurrent.futures
    import queue

    from .io import read_json
    if len(devices) != len(set(devices)) or not devices:
        raise ValueError("Provide distinct devices; each device has one worker")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    entries = read_json(plan)["entries"]
    pending = queue.Queue()
    results = []
    for index, entry in enumerate(entries):
        if entry["state"] == "ready":
            pending.put((index, entry))
        else:
            results.append({**entry, "index": index})

    def worker(device):
        local = []
        while True:
            try:
                index, entry = pending.get_nowait()
            except queue.Empty:
                break
            name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{index:05d}-{entry['method']}-{entry['fm']}-s{entry['seed']}")
            run = root / name
            cmd = [sys.executable, "-m", "milplus", "run", "--data", entry["data"], "--method", entry["method"],
                   "--fm", entry["fm"], "--seed", str(entry["seed"]), "--output", str(run), "--device", device]
            if config:
                cmd += ["--config", str(config)]
            if epochs:
                cmd += ["--epochs", str(epochs)]
            # Existing results are never overwritten, silently resumed or declared equivalent.
            if run.exists():
                local.append({**entry, "index": index, "state": "skipped_existing", "run": str(run)})
                continue
            with open(root / f"{name}.log", "x") as log:
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait(timeout=15)
                    raise
            local.append({**entry, "index": index, "state": "completed" if code == 0 else "failed",
                          "exit_code": code, "device": device, "run": str(run)})
        return local

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for future in [pool.submit(worker, device) for device in devices]:
            results.extend(future.result())
    results.sort(key=lambda r: r["index"])
    write_json(root / "matrix_results.json", {"plan": str(Path(plan).resolve()), "entries": results})
    return {"completed": sum(r["state"] == "completed" for r in results),
            "failed": sum(r["state"] == "failed" for r in results), "total": len(results), "output": str(root)}


def summarize_runs(root, output, subset="test"):
    import pandas as pd

    from .io import read_json
    records = []
    for path in sorted(Path(root).rglob("request.json")):
        req = read_json(path)
        status = read_json(path.parent / "status.json")
        metric_path = path.parent / f"{subset}_metrics.json"
        metrics = read_json(metric_path) if metric_path.exists() else {}
        records.append({"run": str(path.parent), "data": req["data"], "fm": req["fm"], "method": req["method"],
                        "seed": req["seed"], "synthetic": req["synthetic"], "status": status["state"],
                        "evaluated": metric_path.exists(), "wall_seconds": status.get("wall_seconds"), **metrics})
    pd.DataFrame(records).to_csv(output, index=False)
    return {"runs": len(records), "output": str(output), "note": "No pooling across tasks, seeds or metric scales"}
