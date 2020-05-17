"""
Used to Calculate Micro Avg F-Score and (Specificity, Precision at different recall levels by varying probabilities), AUC score
and a combined list of labels cumulating all folds for Cascade and Ensemble-Boolean
Input:
Folder BERT_Output_CV_ITL_Cascade (Cascade Output) OR BERT_Output_CV_ITL_Ensemble_Boolean (Ensemble-Boolean Output)
Output:
Overall_Cross_Validation_Report.txt, Final_Labels.csv, Fold_Wise_Classification_Report.txt
"""

import os
import csv
import numpy as np
from src.Model_Evaluation.specificity import roc_curve_revised
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from src.Model_Evaluation.modified_classification_report import classification_report

path = os.path.abspath("../../Dataset/Classification_Reports/Del_Fiol")
AnswerDir = "Answer_Data"  # Use For Del Fiol
# AnswerDir = "Answer_Data_Full"  # Use For Marshall
itlstepcv = "BERT_Output_CV_ITL_Cascade_PT_Del_Fiol"
# itlstepcv = "BERT_Output_CV_ITL_Ensemble_Boolean_PT"


def ReadAnswerLabels(path, directory, num_files=10):
    """Used to read the Gold Standard Labels to know which combination of Task 1,2,3,4 labels is true according to
    Del Fiol and Marshall Conditions"""
    Answer = []
    for i in range(0, num_files):
        with open(os.path.join(path, directory, "Test_Data_Part_{}_Answer.tsv".format(i))) as fp:
            reader = csv.reader(fp, delimiter='\t')
            for j, row in enumerate(reader):
                if j == 0:
                    continue
                Answer.append([row[0], row[2]])
    return Answer
            

def ReadLabels(path, directory, itl=False):
    task1_labels = []
    task2_labels = []
    task3_labels = []
    task4_labels = []
    files = os.listdir(os.path.join(path, directory))
    for j, file in enumerate(files):
        if(file.startswith('Reqd')):
            k = int(file[-5])
            with open(os.path.join(path, directory, file)) as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    row.insert(1, str(k - 1))
                    if(i == 0):
                        continue
                    if k == 1:
                        task1_labels.append(row)
                    elif k == 2:
                        task2_labels.append(row)
                    elif k == 3:
                        task3_labels.append(row)
                    elif k == 4:
                        task4_labels.append(row)
    return task1_labels, task2_labels, task3_labels, task4_labels

                    
label_converter = {
        0: ["NA", "O",],
        1: ["F", "T"],
        2: ["NA", "TR", ],
        3: ["F", "T"],
    }

task1, task2, task3, task4 = ReadLabels(path, itlstepcv, True)
    
Answer = ReadAnswerLabels(path, AnswerDir, 10)
task1.sort(key=lambda x: x[0])
task2.sort(key=lambda x: x[0])
task3.sort(key=lambda x: x[0])
task4.sort(key=lambda x: x[0])
Answer.sort(key=lambda x: x[0])


def ConvertLabels(labels, num_tasks, label_converter):
    for i in range(num_tasks):
        task_labels = [row for row in labels if row[1] == str(i)]
        for j, row in enumerate(task_labels):
            task_labels[j][2] = label_converter[i][int(task_labels[j][2])]
            task_labels[j][3] = label_converter[i][int(task_labels[j][3])]


ConvertLabels(task1, 5, label_converter)
ConvertLabels(task2, 5, label_converter)
ConvertLabels(task3, 5, label_converter)
ConvertLabels(task4, 5, label_converter)

compute_task2 = []
compute_task3 = []
compute_task4 = []


def DoStepEliminateTask(task1, task2, task3, task4, Answer):
    final_labels = [[] for i in range(len(Answer))]
    final_labels_probs = [[] for i in range(len(task4))]
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    totCount = 0
    for i, row in enumerate(task1):
        totCount += 1
        if(row[3] in ['GM', 'CR', 'R', 'NA']):
            final_labels[i].extend(task4[i][0:2] + [Answer[i][1]] + ['FALSE'])
            final_labels_probs[i].extend(task1[i][0:2] + [1 if Answer[i][1] == 'TRUE' else 0] + [float(task1[i][4])])
            if(Answer[i][1] == 'TRUE'):
                count1 += 1
    
    for i, row in enumerate(task2):
        if(len(final_labels[i]) == 0):
            compute_task2.append(row)
        if(row[3] in ['F'] and len(final_labels[i]) == 0):
            final_labels[i].extend(task4[i][0:2] + [Answer[i][1]] + ['FALSE'])
            final_labels_probs[i].extend(task2[i][0:2] + [1 if Answer[i][1] == 'TRUE' else 0] + [float(task2[i][4])])
            if(Answer[i][1] == 'TRUE'):
                count2 += 1
    
    for i, row in enumerate(task3):
        if(len(final_labels[i]) == 0):
            compute_task3.append(row)
        if(row[3] not in ['TR'] and len(final_labels[i]) == 0):
            final_labels[i].extend(task4[i][0:2] + [Answer[i][1]] + ['FALSE'])
            final_labels_probs[i].extend(task3[i][0:2] + [1 if Answer[i][1] == 'TRUE' else 0] + [float(task3[i][4])])
            if(Answer[i][1] == 'TRUE'):
                count3 += 1
    for i, row in enumerate(task4):
        if(len(final_labels[i]) == 0):
            compute_task4.append(row)
        if(row[3] in ['F'] and len(final_labels[i]) == 0):
            final_labels[i].extend(task4[i][0:2] + [Answer[i][1]] + ['FALSE'])
            final_labels_probs[i].extend(task4[i][0:2] + [1 if Answer[i][1] == 'TRUE' else 0] + [float(task4[i][4])])
            if(Answer[i][1] == 'TRUE'):
                count4 += 1
        elif(row[3] in ['T'] and len(final_labels[i]) == 0):
            final_labels[i].extend(task4[i][0:2] + [Answer[i][1]] + ['TRUE'])
            final_labels_probs[i].extend(task4[i][0:2] + [1 if Answer[i][1] == 'TRUE' else 0] + [float(task4[i][4])])
            if(Answer[i][1] == 'FALSE'):
                count4 += 1
    print("CONFLICTS: \nTot Count: {}\nCount 1: {}\nCount 2: {}\nCount 3: {}\nCount 4: {}".format(
            totCount, count1, count2, count3, count4))
    return final_labels, final_labels_probs


final_labels, final_labels_probs = DoStepEliminateTask(task1, task2, task3, task4, Answer)

file_path = os.path.join(path, itlstepcv, "Overall_Cross_Validation_Report.txt")
file = open(file_path, 'w')


file.write(classification_report([row[2] for row in final_labels], [row[3] for row in final_labels]))
fpr, tpr, thresholds = roc_curve(np.array([int(row[2]) for row in final_labels_probs]),
                                 np.array([float(row[3]) for row in final_labels_probs]), pos_label=1)
file.write("\n\n\nAUC Score: " + str(auc(fpr, tpr)) + "\n\n")
fpr, tpr, thresh = roc_curve_revised(np.array([int(row[2]) for row in final_labels_probs]),
                                     np.array([float(row[3]) for row in final_labels_probs]), pos_label=1)
precision, recall, thresholds = precision_recall_curve([int(row[2]) for row in final_labels_probs],
                                                       [float(row[3]) for row in final_labels_probs], pos_label=1)
for pre, rec, fap, th in zip(precision, recall, fpr, thresholds):
    file.write("Precision: " + str(pre) + "\tRecall: " + str(rec) + "\tSpecificity: " +
               str(1-fap) + "\tThreshold: " + str(th) + "\n")

prob_dict = dict()
for row in final_labels_probs:
    prob_dict[row[0]] = row[3]
for i, row in enumerate(final_labels):
    final_labels[i].append(prob_dict[row[0]])
with open(os.path.join(path, itlstepcv, "Final_Labels.csv"), 'w') as fp:
    writer = csv.writer(fp)
    for row in final_labels:
        writer.writerow(row)
file.close()

file_path = os.path.join(path, itlstepcv, "Task Wise Report.txt")

file = open(file_path, 'w')
file.write("\n\nTask 1: \n" + classification_report([row[2] for row in task1], [row[3] for row in task1]) + "\n\n")
file.write("\n\nTask 2: \n" + classification_report([row[2] for row in compute_task2], [row[3] for row in compute_task2]) + "\n\n")
file.write("\n\nTask 3: \n" + classification_report([row[2] for row in compute_task3], [row[3] for row in compute_task3]) + "\n\n")
file.write("\n\nTask 4: \n" + classification_report([row[2] for row in compute_task4], [row[3] for row in compute_task4]) + "\n\n")
file.close()
