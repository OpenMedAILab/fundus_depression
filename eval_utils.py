# eval_utils.py
import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib.pyplot as plt


@torch.no_grad()
def predict_patient_level(
    model,
    loader,
    device,
    pool="mean",          # "mean" or "max"
    use_sigmoid=True,     # binary logit -> prob
):
    """
    返回：
      df_patient: columns = [eid, y_true, y_prob]
      y_true (np.array), y_prob (np.array)
    """
    model.eval()

    eid_scores = defaultdict(list)
    eid_label = {}

    for imgs, ys, eids in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs).squeeze(-1)

        if use_sigmoid:
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        else:
            probs = logits.detach().cpu().numpy()

        ys = ys.detach().cpu().numpy()

        for p, y, eid in zip(probs, ys, eids):
            eid_scores[eid].append(float(p))
            y_int = int(y)
            if eid in eid_label and eid_label[eid] != y_int:
                raise ValueError(f"Label conflict for eid={eid}: {eid_label[eid]} vs {y_int}")
            eid_label[eid] = y_int

    rows = []
    for eid, ps in eid_scores.items():
        if pool == "max":
            s = float(np.max(ps))
        else:
            s = float(np.mean(ps))
        rows.append((eid, eid_label[eid], s))

    df = pd.DataFrame(rows, columns=["eid", "y_true", "y_prob"])
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)

    y_true = df["y_true"].to_numpy(dtype=int)
    y_prob = df["y_prob"].to_numpy(dtype=float)
    return df, y_true, y_prob


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    """
    返回 dict，包含 AUC/AP/ACC/Precision/Recall(Sen)/Spe/F1 以及 confusion matrix
    """
    y_pred = (y_prob >= threshold).astype(int)

    # AUC/AP
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    ap  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    # basic metrics
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # sensitivity
    f1  = f1_score(y_true, y_pred, zero_division=0)

    # confusion matrix & specificity
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "auc_roc": float(auc),
        "auc_pr": float(ap),
        "acc": float(acc),
        "precision": float(pre),
        "recall_sen": float(rec),
        "specificity_spe": float(spe),
        "f1": float(f1),
        "cm": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        },
        "pos_rate": float(np.mean(y_true)),
        "n_patients": int(len(y_true)),
    }


def plot_roc(y_true, y_prob, out_png, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (AUC={auc:.4f})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_pr(y_true, y_prob, out_png, title="PR Curve"):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} (AP={ap:.4f})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_confusion_matrix(cm_dict, out_png, title="Confusion Matrix"):
    # cm order: tn fp / fn tp
    cm = np.array([[cm_dict["tn"], cm_dict["fp"]],
                   [cm_dict["fn"], cm_dict["tp"]]], dtype=int)

    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xticks([0,1], ["Pred 0", "Pred 1"])
    plt.yticks([0,1], ["True 0", "True 1"])

    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def evaluate_and_save(
    model,
    loader,
    device,
    out_dir,
    prefix="test",
    threshold=0.5,
    pool="mean",
):
    """
    一站式：预测 -> 指标 -> 图 -> 导出
    输出：metrics dict
    """
    os.makedirs(out_dir, exist_ok=True)

    df, y_true, y_prob = predict_patient_level(model, loader, device, pool=pool, use_sigmoid=True)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)

    # save predictions
    pred_csv = os.path.join(out_dir, f"{prefix}_pred_patient.csv")
    df.to_csv(pred_csv, index=False)

    # save metrics
    metrics_json = os.path.join(out_dir, f"{prefix}_metrics.json")
    save_json(metrics, metrics_json)

    # plots
    roc_png = os.path.join(out_dir, f"{prefix}_roc.png")
    pr_png  = os.path.join(out_dir, f"{prefix}_pr.png")
    cm_png  = os.path.join(out_dir, f"{prefix}_cm.png")

    if len(np.unique(y_true)) > 1:
        plot_roc(y_true, y_prob, roc_png, title=f"{prefix} ROC")
        plot_pr(y_true, y_prob, pr_png, title=f"{prefix} PR")

    plot_confusion_matrix(metrics["cm"], cm_png, title=f"{prefix} CM (thr={threshold})")

    # print paths (optional)
    metrics["_artifacts"] = {
        "pred_csv": pred_csv,
        "metrics_json": metrics_json,
        "roc_png": roc_png,
        "pr_png": pr_png,
        "cm_png": cm_png,
    }
    return metrics


def plot_domain_shift_curve(domain_metrics, out_png, metric_key="auc_roc", title=None):
    """
    domain_metrics: list of dicts, each dict needs:
      {"domain": "UKB", "auc_roc": 0.62, "auc_pr": 0.15, ...}
    """
    domains = [d["domain"] for d in domain_metrics]
    vals = [d.get(metric_key, np.nan) for d in domain_metrics]

    plt.figure()
    plt.plot(domains, vals, marker="o")
    plt.ylabel(metric_key)
    plt.xlabel("Domain")
    plt.title(title or f"Domain shift: {metric_key}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
