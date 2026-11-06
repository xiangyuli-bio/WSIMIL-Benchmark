"""Complete pinned MR-Block recipes, retaining five MR and five baseline models."""

RECIPE = {
    "epochs": 200, "batch_size": 1, "optimizer": "AdamW", "lr": 5e-4,
    "weight_decay": 1e-5, "dropout": .25, "size_arg": "small", "rank": 64,
    "scheduler": "LinearLR(start_factor=.01,end_factor=1,total_iters=200), after each update",
    "early_stop_patience": 20, "early_stop_after_epoch_zero_based": 50,
    "early_stop_selection": "minimum VAL CE, equal loss also updates selected weights",
    "final_selection": "last weights after final VAL, as original early_stopping=False",
    "weighted_sample": False, "workers": 0, "patch_cap": None, "amp": False,
}
MODELS = ("mr-abmil", "mr-transmil", "mr-dgrmil", "mr-rrtmil", "mr-cate",
          "abmil", "transmil", "dgrmil", "rrtmil", "cate")
DEFAULT_PROFILE = "mrblock_author_200_mr_abmil_earlystop_locked_test"
PROFILES = [
    {
        "id": f"mrblock_author_200_{model.replace('-', '_')}_{selector}_locked_test",
        "family": "mrblock",
        "recipe": (
            f"pinned native {model}; AdamW5e-4/wd1e-5/LinearLR200updates; "
            f"full200epoch budget and original {selector} selector; merge-aware exact recovery"
        ),
        "tasks": ["classification"],
        "required_inputs": ["patch_features", "single_slide_bag"] + (
            ["aligned_original_concept_features"] if model.endswith("cate") else []),
        "class_count": None, "external_source": "mrblock",
        "native_recipe": {**RECIPE, "model_type": model, "early_stopping": selector == "earlystop"},
        "comparison_role": "MR_module" if model.startswith("mr-") else "author_baseline_control",
        "status": "adapted_under_verification", "gpu_validation": "not_run",
        "real_wsi_validation": "not_run", "paper_reproduction": "not_run",
        "adaptations": [
            "patient-disjoint task manifest fixes split membership; few-shot eligibility is explicit input",
            "TEST is locked; original train_fold constructs TEST loader and evaluates it after fitting",
            "workers0, explicit runtime dependency pins and full numerical identity",
            "immutable best and full model/AdamW/scheduler/sampler/RNG/minibatch recovery",
            "native merged flags and module modes restored to avoid fresh-reload double merging",
        ],
        "native_behaviors_preserved": [
            "original MR initialization, trainable bias, frozen base weight and rank64 nonlinear path",
            "original eval linear merge delta even though training residual contains GELU",
            "original CATE stochastic reparameterization even during VAL/evaluation",
            "original MR-CATE trainer uses CE only; discarded IB losses are not silently added",
            "original DGR variant updates centers from predicted class during TRAIN",
        ],
    }
    for model in MODELS
    for selector in ("earlystop", "final")
]
