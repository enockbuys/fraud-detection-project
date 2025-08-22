import numpy as np
def classification_summary(true_labels, predicted_labels):
    true_positives = np.sum((true_labels == 1) & (predicted_labels == 1))
    true_negatives = np.sum((true_labels == 0) & (predicted_labels == 0))
    false_positives = np.sum((true_labels == 0) & (predicted_labels == 1))
    false_negatives = np.sum((true_labels == 1) & (predicted_labels == 0))

    total_samples = len(true_labels)

    if (true_positives + false_positives) == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)

    if (true_positives + false_negatives) == 0:
        recall = 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)

    if (precision + recall) == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    accuracy = (true_positives + true_negatives) / total_samples

    print("Classification Summary:")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1_score, 4))
    print("Confusion Matrix:")
    print("True Positives:", true_positives)
    print("False Positives:", false_positives)
    print("False Negatives:", false_negatives)
    print("True Negatives:", true_negatives)