import numpy as np


class Metrics:
    def __init__(self, num_classes):
        self.conf_matrix = np.zeros((num_classes, num_classes))
        self.TP = []
        self.FP = []
        self.TN = []
        self.FN = []
        self.calculated_accuracy = []
        self.calculated_precision = []
        self.calculated_recall = []
        self.calculated_specificity = []
        self.calculated_f1 = []
        self.auc_roc = 0