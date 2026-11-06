"""Author ASMIL recipes; the older controlled ``asmil`` port stays explicit."""

DIMENSIONS = {384: 128, 512: 256, 768: 384, 1024: 512, 1536: 768}
PROFILES = [
    {
        "id": f"asmil_author_{epochs}_{mode}_locked_test",
        "family": "asmil",
        "recipe": (
            f"original ACMIL_MYMHA / independently initialized train-mode teacher; "
            f"full {epochs} epochs, batch1/full bags, AdamW/cosine/clip5, "
            f"{mode} teacher update before optimizer, micro-F1+macro-AUROC VAL selection; "
            "explicit fixed-patient, seed and locked-test repairs"
        ),
        "tasks": ["classification"], "required_inputs": ["patch_features"],
        "class_count": None, "external_source": "asmil",
        "gpu_validation": "not_run", "real_wsi_validation": "not_run",
        "status": "adapted_under_verification", "paper_reproduction": "not_run",
        "native_recipe": {
            "epochs": epochs, "batch_size": 1, "patch_cap": None,
            "lr": 1e-4, "initial_optimizer_lr": .001, "weight_decay": 1e-5,
            "warmup_epoch": 0, "min_lr": 0, "clip_grad_norm": 5.,
            "ema_decay": .999, "n_token": 1, "n_drop": 0, "temp": 1.,
            "early_stopping": None,
        },
        "author_config": (
            "config/camelyon_config.yml" if epochs == 50
            else "config/camelyon_medical_ssl_config.yml"
        ),
        "teacher_update": mode,
        "repairs": [
            "patient-disjoint fixed task manifest instead of source slide split",
            "explicit unified seed; source main comments out set_seed",
            "locked TEST; source evaluates TEST each epoch",
            "immutable best snapshot; complete online/teacher/optimizer/RNG/sampler recovery",
        ] + (["update finite states instead of source NaN-only condition"]
             if mode == "finite_ema_repaired" else []),
        "metric_runtime": "torchmetrics==1.3.0.post0 explicit reference, source does not pin version",
    }
    for epochs in (50, 100)
    for mode in ("literal_nan_teacher", "finite_ema_repaired")
]

DEFAULT_PROFILE = "asmil_author_50_literal_nan_teacher_locked_test"
