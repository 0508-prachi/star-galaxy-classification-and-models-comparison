from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score
astronomy_cm = confusion_matrix(y_true, y_pred_classes)
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
astronomy_confusion_matrix = confusion_matrix(y_test, astronomy_y_pred)
print("RF Model Confusion Matrix: \n", astronomy_confusion_matrix)
sns.heatmap(astronomy_confusion_matrix, annot=True, fmt='d', cmap="Blues")
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,