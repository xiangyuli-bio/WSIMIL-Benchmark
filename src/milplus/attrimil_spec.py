"""Complete original AttriMIL SGD/spatial/ranking recipes with explicit repairs."""

RECIPE = {
    "epochs": 200, "batch_size": 1, "lr": 2e-4, "momentum": .9,
    "weight_decay": 1e-5, "spatial_weight": 1., "rank_weight": 5.,
    "kernel_size": 3, "queue_size": 4, "queue_reset": "every epoch",
    "train_sampling": "class-balanced WeightedRandomSampler with replacement",
    "scheduler": None, "gradient_clipping": None, "patch_cap": None,
    "initial_best_loss": 10000., "patience": 20,
    "early_stop_condition": "retain >20 and zero-based epoch >50",
}
DEFAULT_PROFILE = "attrimil_author_200_main_repaired"
PROFILES = [
    {
        "id": name, "family": "attrimil",
        "recipe": (
            "original AttriMIL class-specific architecture, SGD2e-4/momentum.9/wd1e-5, "
            "spatial1/rank5, actual3x3 neighbors, epoch-local queues4, class-balanced "
            "replacement sampling, complete200epoch maximum, VAL CE selector; "
            + ("original main no early stopping" if not early else "original function-default early stopping")
            + "; disclosed spatial-axis/logging/device/prediction repairs"
        ),
        "tasks": ["classification"],
        "required_inputs": ["patch_features", "per_slide_coords", "spatial_metadata", "single_slide_bag"],
        "class_count": None, "external_source": "attrimil",
        "native_recipe": {**RECIPE, "early_stopping": early},
        "gpu_validation": "not_run", "real_wsi_validation": "not_run",
        "status": "adapted_under_verification", "paper_reproduction": "not_run",
        "repairs": [
            "spatial_constraint score=A[0,c] instead of A[:,c] for original [1,C,N] attributes",
            "remove unused undefined loss_ins logging reference; original objective unchanged",
            "allocate original A_raw on feature device; CPU semantics unchanged",
            "evaluate original5-output AttriMIL probabilities instead of broken DSMIL-style summary",
            "fixed patient-disjoint task manifest and explicit seed",
            "immutable best snapshots and complete SGD/queues/RNG/weighted-sampler recovery",
        ],
    }
    for name, early in [
        (DEFAULT_PROFILE, False),
        ("attrimil_author_200_function_earlystop_repaired", True),
    ]
]
