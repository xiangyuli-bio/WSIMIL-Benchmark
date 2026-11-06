"""Standard feature input with literal native per-index patch selection."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SlideBags(Dataset):
    def __init__(self, bundle, subset, config):
        self.bundle, self.subset, self.config = bundle, subset, config
        self.rows = bundle.records(subset)
        self.num_classes = len(bundle.classes)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        features, coords = self.bundle.load(row)
        if coords is None:
            coords = torch.zeros(len(features), 2, dtype=torch.float32)
        count = actual = len(features)
        if self.subset == "train":
            maximum = self.config["max_seq_length"]
            if count > maximum:
                selected = np.random.RandomState(42 + index).choice(count, maximum, replace=False)
                features, coords, actual = features[selected], coords[selected], maximum
            elif count < maximum:
                features = torch.cat([features, torch.zeros(maximum - count, features.shape[1])])
                coords = torch.cat([coords, torch.zeros(maximum - count, coords.shape[1])])
        return (features, coords, torch.tensor(actual, dtype=torch.long),
                torch.tensor(int(row["target"]), dtype=torch.long), row["slide_id"],
                self.bundle.config.get("dataset_id", "milplus_task"))


def train_only_plan(bundle, metric, author, seed):
    rows = bundle.records("train")
    counts = [len(bundle.load(row)[0]) for row in rows]
    maximum = int(np.median(counts) * .5)
    if maximum < 1:
        raise ValueError("nnMIL original half-median sampling would make empty TRAIN bags")
    statistics = {
        "feature_dimension": bundle.dim,
        "num_patches_per_slide": {
            "min": int(min(counts)), "max": int(max(counts)), "mean": float(np.mean(counts)),
            "median": float(np.median(counts)), "percentile_25": float(np.percentile(counts, 25)),
            "percentile_75": float(np.percentile(counts, 75)),
            "percentile_95": float(np.percentile(counts, 95)),
        },
        "recommended_max_seq_length": maximum,
    }
    # Use the literal original config generator with only TRAIN rows supplied.
    planner = author.planner.ExperimentPlanner.__new__(author.planner.ExperimentPlanner)
    planner.task_type, planner.evaluation_setting = "classification", "official_split"
    planner.dataset_info = {"metric": metric, "labels": {name: i for i, name in enumerate(bundle.classes)}}
    planner.df = pd.DataFrame([
        {"slide_id": row["slide_id"], "patient_id": row["global_patient_id"],
         "label": int(row["target"]), "split": "train"} for row in rows])
    config = planner.generate_training_config(statistics)
    return {
        "task_type": "classification", "task_name": bundle.config.get("dataset_id", "milplus_task"),
        "feature_dir": str(bundle.root), "metric": metric, "evaluation_setting": "official_split",
        "labels": planner.dataset_info["labels"], "training_configuration": config,
        "feature_statistics": statistics, "random_seed": seed,
        "planner_scope": "TRAIN-only repair; no VAL/TEST shapes or labels contribute",
        "planning_slide_ids": [row["slide_id"] for row in rows],
    }
