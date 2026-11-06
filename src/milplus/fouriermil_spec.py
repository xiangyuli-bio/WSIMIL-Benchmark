"""The fixed author Fourier-MIL recipe; reference precision is explicit."""

SOURCE_ID = "fouriermil_author"
SOURCE_COMMIT = "0192fe7f7a4fdbb4161148d0b88b0d67dbaa14f5"
BACKEND = "fouriermil_original_200_accumulation2_val_loss_locked_test"
RECIPE = {
    "epochs": 200, "patience": 10, "min_delta": 0.,
    "batch_size": 1, "workers": 8, "accumulate_grad_batches": 2,
    "input_dimension": 1024, "hidden_dimension": 512, "blocks": 2,
    "dropout": 0., "opt": "lookahead_radam", "lr": 2e-4,
    "opt_eps": None, "opt_betas": None, "momentum": None, "weight_decay": 1e-5,
    "selection": "strict minimum literal author VAL log(softmax) loss",
    "data_shuffle": False, "train_shuffle": True, "seed_in_author_yaml": 2021,
    "gradient_clipping": None, "scheduler": None,
    "original_environment": "torch1.7.1+cu110 / pytorch_lightning1.2.3",
    "compatibility_changes": [
        "explicit caller device replaces original CLS-token cuda()",
        "patient-disjoint Bundle replaces author cohort CSV and absolute feature paths",
        "unchanged author model, initialization, training_step, validation_step and VAL loss",
        "explicit current-framework accumulation and native AMP equivalent control flow",
        "zero_grad(set_to_none=False) retains torch1.7 default",
        "empty current-Optimizer hook registries repair original Lookahead zero_grad/step interface",
        "named fast/slow/cache optimizer restoration repairs incompatible original loader",
        "TEST locked until a separately requested evaluation after complete fitting",
    ],
}
PROFILES = [
    {
        "id": f"fouriermil_author_200_{precision}_val_loss_locked_test",
        "family": "fouriermil",
        "recipe": f"original Fourier1024->512, Lookahead-RAdam, accumulation2, {precision}, VAL-loss patience10",
        "tasks": ["classification"], "required_inputs": ["patch_features", "single_slide_bag"],
        "class_count": None, "external_source": SOURCE_ID,
        "author_recipe": {**RECIPE, "precision": 16 if precision == "fp16" else 32},
        "comparison_role": ("native_CUDA_FP16_current_framework_compatibility"
                            if precision == "fp16" else "explicit_CPU_float32_reference"),
        "status": "adapted_under_verification", "gpu_validation": "not_run",
        "real_wsi_validation": "not_run", "paper_reproduction": "not_run",
    }
    for precision in ("fp16", "cpu32_reference")
]
