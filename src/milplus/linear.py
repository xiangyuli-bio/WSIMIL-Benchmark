"""Raw-feature means, train-only standardization, multinomial logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .io import write_json
from .metrics import classification_metrics


def matrix(bundle, subset):
    from .engine import units
    xs, rows = [], []
    for group in units(bundle, subset):
        bags = [bundle.load(row)[0].numpy() for row in group]
        xs.append(sum(b.sum(0, dtype=np.float64) for b in bags) / sum(len(b) for b in bags))
        row = dict(group[0])
        if len(group) > 1:
            row["slide_id"] = "patient:" + row["global_patient_id"]
        rows.append(row)
    return np.stack(xs), rows


def fit_linear(bundle, output, config):
    if config["model"]:
        allowed = {"C", "max_iter"}
        if set(config["model"]) - allowed:
            raise ValueError("mean_linear model options: C, max_iter")
    x, rows = matrix(bundle, "train")
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=config["model"].get("C", 1.0),
                               max_iter=config["model"].get("max_iter", 1000), random_state=0)
    model.fit(scaler.transform(x), [r["target"] for r in rows])
    np.savez(output / "linear.npz", mean=scaler.mean_, scale=scaler.scale_, coef=model.coef_,
             intercept=model.intercept_, classes=model.classes_)
    predictions = predict_linear(bundle, output, "val")
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    write_json(output / "validation_metrics.json", classification_metrics(
        predictions, len(bundle.classes), bundle.config.get("evaluation_unit", "slide")))


def predict_linear(bundle, output, subset):
    from scipy.special import expit, softmax
    x, rows = matrix(bundle, subset)
    with np.load(output / "linear.npz", allow_pickle=False) as weights:
        z = ((x - weights["mean"]) / weights["scale"]) @ weights["coef"].T + weights["intercept"]
        p = np.column_stack([1 - expit(z[:, 0]), expit(z[:, 0])]) if z.shape[1] == 1 else softmax(z, axis=1)
        if not np.array_equal(weights["classes"], np.arange(len(bundle.classes))):
            raise ValueError("Class mapping mismatch")
    output_rows = []
    for row, values in zip(rows, p):
        result = {k: row[k] for k in ("slide_id", "patient_id", "global_patient_id", "target", "subset")}
        result.update({f"p_{i}": float(v) for i, v in enumerate(values)})
        output_rows.append(result)
    return pd.DataFrame(output_rows)
