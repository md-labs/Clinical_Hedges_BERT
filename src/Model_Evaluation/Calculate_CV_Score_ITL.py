"""
Used to Calculate Micro Avg F-Score and (Specificity, Precision at different recall levels by varying probabilities), AUC score
and a combined list of labels cumulating all folds for ITL and Ensemble
Input:
Folder BERT_Output_CV_ITL (ITL Output) OR BERT_Output_CV_Ensemble (Ensemble Output)
Output:
Overall_Cross_Validation_Report.txt, Final_Labels.csv, Fold_Wise_Classification_Report.txt
"""

import os
import csv
from src.Model_Evaluation.modified_classification_report import classification_report
from src.Model_Evaluation.specificity import roc_curve_revised
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix
import numpy as np

path = os.path.abspath("../../Dataset/Classification_Reports/Del_Fiol")
itlcv = "BERT_Output_CV_ITL_1.5_1.5_PT_Del_Fiol"
cvensemble = "BERT_Output_CV_Ensemble"


def CVScore(path, directory):
    files = os.listdir(os.path.join(path, directory))
    Cfile = open(os.path.join(path, directory, "Fold_Wise_Classification_Report.txt"), 'w')
    labels = []
    for i, file in enumerate(files):
        curLabel = []
        if(file.startswith('Reqd')):
            with open(os.path.join(path, directory, file)) as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if(i == 0):
                        continue
                    labels.append(row)
                    curLabel.append(row)
            Cfile.write("Classification Report for Fold: {}\n".format(file[17:-4]))
            Cfile.write(classification_report([row[1] for row in curLabel], [row[2] for row in curLabel]) + "\n\n")

    with open(os.path.join(path, directory, "Overall_Cross_Validation_Report.txt"), 'w') as fp:
        fp.write(classification_report([row[1] for row in labels], [row[2] for row in labels]))
        fpr, tpr, thresh = roc_curve(np.array([int(row[1]) for row in labels]), np.array([float(row[3]) for row in labels]), pos_label=1)
        fp.write("\n\n\nAUC Score: " + str(auc(fpr, tpr)) + "\n\n")
        fpr, tpr, thresh = roc_curve_revised(np.array([int(row[1]) for row in labels]),
                                            np.array([float(row[3]) for row in labels]), pos_label=1)
        precision, recall, thresholds = precision_recall_curve([int(row[1]) for row in labels], [float(row[3]) for row in labels], pos_label=1)
        for pre, rec, fap, th in zip(precision, recall, fpr, thresholds):
            fp.write("Precision: " + str(pre) + "\tRecall: " + str(rec) + "\tSpecificity: " +
                     str(1-fap) + "\tThreshold: " + str(th) + "\n")
    Cfile.close()
    with open(os.path.join(path, directory, "Final_Labels.csv"), 'w') as fp:
        writer = csv.writer(fp)
        for row in labels:
            writer.writerow(row)


# CVScore(path, cvensemble)
CVScore(path, itlcv)
