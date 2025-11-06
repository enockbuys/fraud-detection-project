import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def classification_summary(true_labels, predicted_labels, pred_probs=None):
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    pred_probs = np.asarray(pred_probs) if pred_probs is not None else None

    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length")
    if not np.all(np.isin(true_labels, [0, 1])) or not np.all(np.isin(predicted_labels,[0, 1])):
        raise ValueError("Labels must be binary (0 or 1)")

    true_labels = np.array(true_labels, dtype=np.int64)
    predicted_labels = np.array(predicted_labels, dtype=np.int64)

    # Confusion matrix elements
    tp = np.sum((true_labels == 1) & (predicted_labels == 1))
    tn = np.sum((true_labels == 0) & (predicted_labels == 0))
    fp = np.sum((true_labels == 0) & (predicted_labels == 1))
    fn = np.sum((true_labels == 1) & (predicted_labels == 0))

    # Basic metrics
    accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn)
    }
    # AUC metrics if probabilities are provided
    if pred_probs is not None:
        try:
            # Handle different probability formats
            if pred_probs.ndim > 1:
                # If it's a 2D array, take probabilities for class 1
                if pred_probs.shape[1] == 2:
                    pred_probs_1d = pred_probs[:, 1]  # Probability of class 1
                else:
                    pred_probs_1d = pred_probs[:, 0]  # Fallback to first column
            else:
                pred_probs_1d = pred_probs
            metrics['roc_auc'] = roc_auc_score(true_labels, pred_probs_1d)
            metrics['pr_auc'] = average_precision_score(true_labels, pred_probs_1d)
        except Exception as e:
            print(f"Error computing AUC metrics: {e}")
            metrics['roc_auc'] = 0
            metrics['pr_auc'] = 0
    return metrics