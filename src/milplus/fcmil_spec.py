"""Original FC-MIL mechanism with explicitly chosen benchmark training budgets."""

RECIPE = {
    "epochs": 100, "lr": 2e-4, "batch_size": 1, "workers": 0,
    "input_dimension": 1024, "hidden_dimension": 512,
    "topk_ratio": .03, "lam": .5, "weight_decay": 1e-5,
    "scheduler_factor": .1, "scheduler_patience": 5,
    "selection": "strict real VAL macro AUROC improvement, starting at0 as native manager",
    "scheduler_order": "after validation and selection, native ReduceLROnPlateau on VAL CE",
    "provenance": {
        "author_code": [
            "original FCMIL/FAA/DAB bodies and exact frequency-attention intervention",
            "native CE+alpha_cr*(-lam*KL_drop+(1-lam)*KL_replace) with detached original probabilities",
            "native Adam wd1e-5 and ReduceLROnPlateau factor.1/patience5",
            "native strict AUC improvement selector with initial best_auc0",
            "source fallback topk_ratio.03, lam.5, alpha_cr.1",
        ],
        "benchmark_choices_missing_from_fixed_author_tree": [
            "100 complete epochs", "Adam lr2e-4", "single-bag batch1",
            "workers0", "seed2026 default; explicit full RNG seed binding",
            "fixed deterministic-algorithm/highest-float32 policy across library and fresh CLI",
        ],
        "repairs": [
            "original dataset logs before logger initialization; reference initializes logger first",
            "manager passes writer to a function without that keyword; logging argument ignored",
            "VAL macro AUROC computed from actual labels and predictions; original return lacks auc",
            "owned output paths replace destructive shared ./checkpoints and ./logs paths",
            "caller device, patient-disjoint Bundle, locked TEST and immutable resumable state",
        ],
    },
}

DEFAULT_PROFILE = "fcmil_original_benchmark100_cr_auc_repaired"
PROFILES = [
    {
        "id": f"fcmil_original_benchmark100_{suffix}_auc_repaired",
        "family": "fcmil",
        "recipe": (
            "original FC-MIL1024->512 spatial/FFT attention; explicit benchmark100epoch "
            f"Adam2e-4 batch1; alpha_cr={alpha}; real VAL AUC and writer-interface repairs"
        ),
        "tasks": ["classification"], "required_inputs": ["patch_features", "single_slide_bag"],
        "class_count": None, "external_source": "fcmil",
        "benchmark_recipe": {**RECIPE, "alpha_cr": alpha},
        "comparison_role": "full_original_mechanism" if alpha else "explicit_no_CR_ablation",
        "author_training_configuration_recovered": False,
        "status": "adapted_under_verification", "gpu_validation": "not_run",
        "real_wsi_validation": "not_run", "paper_reproduction": "not_run",
    }
    for suffix, alpha in [("cr", .1), ("ce_control", 0.)]
]
