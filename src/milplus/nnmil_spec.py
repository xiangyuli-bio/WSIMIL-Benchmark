"""Explicit original classification recipes with TRAIN-only planning repair."""

METRICS = ("auc", "bacc", "f1", "kappa")
DEFAULT_PROFILE = "nnmil_author_100_bacc_train_planned_locked_test"
PROFILES = [
    {
        "id": f"nnmil_author_100_{metric}_train_planned_locked_test",
        "family": "nnmil",
        "recipe": (
            f"native SimpleMIL/AdamW/bfloat16/100epoch maximum/{metric} early stop; "
            "original adaptive plan restricted to TRAIN; native workers4 and exact iterator replay"
        ),
        "tasks": ["classification", "ordinal"],
        "required_inputs": ["patch_features", "patient_disjoint_splits"],
        "class_count": None, "external_source": "nnmil",
        "native_recipe": {
            "metric": metric, "epochs": 100, "patience": 10,
            "planner_scope": "TRAIN features and TRAIN labels only, explicit repair",
            "hidden": "max(256,D//4)", "batch_size": "original adaptive16..48",
            "AMP": "CPU/CUDA bfloat16; original CUDA GradScaler",
            "optimizer": "original AdamW two groups; no decay for gain/bias parameters",
            "scheduler": "original cosine/warmup AFTER each optimizer update, including step0",
            "workers": 4, "dataset_sampling_seed": 42, "use_original_length": False,
        },
        "status": "adapted_under_verification", "gpu_validation": "not_run",
        "real_wsi_validation": "not_run", "paper_reproduction": "not_run",
        "adaptations": [
            "TRAIN-only feature statistics and label proportions; TEST cannot change adaptive plan",
            "standard Bundle replaces CSV/H5 input; identical per-index42 patch selection and padding",
            "absent coordinates use zeros only for unused native loader field; no spatial claim",
            "fixed TEST; serialized best weights and full optimizer/scaler/RNG/sampler recovery",
            "actual original workers4 retained; resume replays deterministic input iteration, never optimizer updates",
            "ordinal endpoint uses classification trainer and explicit kappa selection",
            "torch2.5.1 runtime is a benchmark choice; original environment lists torch2.3.0",
        ],
        "native_behaviors_preserved": [
            "randperm consumed during TRAIN even when hidden>=input dimension",
            "overlapping deterministic chunk ensemble and ddof1 uncertainty, including one-chunk NaN",
            "fixed per-index42 train crop, full original VAL/TEST bags, native padding mask",
            "AUC low-variance sampler, balanced sampler or dedicated-generator random sampler",
            "original strict-improvement early stopping; metric ties increment patience",
            "original shallow in-memory best-state alias; immutable serialized selection used",
        ],
    }
    for metric in METRICS
]
