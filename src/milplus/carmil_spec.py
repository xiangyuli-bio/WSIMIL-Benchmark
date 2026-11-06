"""Pinned CAR-MIL native backbones and complete100epoch locked-TEST recipes."""

RECIPE = {
    "epochs": 100, "batch_size": 1, "optimizer": "Adam",
    "lr": 2e-4, "weight_decay": 1e-5, "dropout": .25, "act": "relu",
    "alpha_effect": .2, "alpha_att": .2, "loss": "ce",
    "scheduler": "CosineAnnealingLR(T_max=100, eta_min=0), after each full TRAIN epoch",
    "fixed_loader_seed": 7784414403328510413,
    "selection": "VAL AUC before TRAIN, strict improvement over initial0",
    "early_stopping": {"patience": 30, "stop_epoch": 130, "save_best_model_stage": 0.},
    "gradient_clipping": None, "patch_cap": None, "amp": False,
}
DEFAULT_PROFILE = "carmil_author_100_attmil_l1_locked_test_repaired"
PROFILES = [
    {
        "id": f"carmil_author_100_{backbone}_{distance.lower()}_locked_test_repaired",
        "family": "carmil",
        "recipe": (
            f"original {backbone} factual/CF network and {distance} objective; full100epochs "
            "Adam2e-4/wd1e-5/cosine100; original pre-TRAIN VAL AUC selector and private "
            "shuffle generator; locked TEST, disclosed NumPy2/CPU timing repairs"
        ),
        "tasks": ["classification"], "required_inputs": ["patch_features", "single_slide_bag"],
        "class_count": 2, "external_source": "carmil",
        "native_recipe": {**RECIPE, "backbone": backbone, "reg_dist": distance,
                          "cls_alpha": .5 if backbone == "dsmil" else 1.,
                          "aux_alpha": .5 if backbone == "dsmil" else 1.},
        "status": "adapted_under_verification", "gpu_validation": "not_run",
        "real_wsi_validation": "not_run", "paper_reproduction": "not_run",
        "repairs": [
            "original EarlyStopping np.Inf becomes np.inf for NumPy2",
            "CPU timing instrumentation omitted; native forward/backward/scheduler unchanged",
            "fixed patient-disjoint TRAIN/VAL and locked TEST replace original default TEST-as-VAL",
            "num_workers0 replaces original2 for exact recorded/recoverable loading",
            "immutable selected state and Adam/scheduler/RNG/private-generator/minibatch recovery",
        ],
        "native_attention_shape": (
            "literal original flattened2N TransMIL CF attention; trainer indexes scalar entries; no stack repair"
            if backbone == "transmil" else "literal original two-row factual/CF attention"
        ),
    }
    for backbone in ("attmil", "gattmil", "transmil", "dsmil")
    for distance in ("L1", "COS")
]
