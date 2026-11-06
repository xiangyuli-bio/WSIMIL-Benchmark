"""Full author PNEA recipes, separate from the existing controlled adaptation."""

RECIPE = {
    "max_epochs": 1000, "batch_size": 8, "train_patches": 1024,
    "evidence_axes": 64, "lambda_l1": 5.0, "lr": 1e-4,
    "weight_decay": 0.0, "patience": 10, "relative_improvement": 0.0005,
    "optimizer": "author_Adam_grouped_bias_no_decay",
    "gradient_clipping": None, "scheduler": None,
}
SOURCE_PROFILE = "pneamil_author_train_selector_snapshot_repaired"
VAL_PROFILE = "pneamil_author_val_selector_repaired"
PROFILES = [
    {
        "id": name, "family": "pneamil",
        "recipe": description, "tasks": ["classification"],
        "required_inputs": ["patch_features"], "class_count": None,
        "external_source": "pneamil", "gpu_validation": "not_run",
        "real_wsi_validation": "not_run", "status": "adapted_under_verification",
        "paper_reproduction": "not_run", "native_recipe": dict(RECIPE),
        "selection_population": population,
        "repairs": ["immutable selected checkpoint",
                    "patient-disjoint externally fixed task split"] + (
                        ["use full validation population instead of source train_loader"]
                        if population == "val" else []),
    }
    for name, population, description in [
        (SOURCE_PROFILE, "train",
         "actual author PNEA model and CE+5meanReLU(W), grouped Adam, "
         "batch8/1024replacement patches/max1000epochs, literal TRAIN-loader "
         "relative-loss selector; explicitly repaired checkpoint snapshot"),
        (VAL_PROFILE, "val",
         "actual author PNEA model and complete original1000/batch8/1024/Adam "
         "recipe with separately named full-VAL relative-loss selector and "
         "immutable selected checkpoint repairs"),
    ]
]
