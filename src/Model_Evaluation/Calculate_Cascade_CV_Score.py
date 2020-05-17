import os
import csv
import numpy as np

from src.Model_Evaluation.modified_classification_report import classification_report

mtl = False
itlanswerflag = True
path = "C:/Clinical_Hedges/Dataset/Classification_Reports/"
AnswerDir = "Answer_Data"
mtlstep = "BERT_Output_CV_MTL"
#itlstep = "BERT_Output_ITL_Step"
#itlstepcv = "BERT_Output_CV_ITL_Min_Cascade"
itlstepcv = "BERT_Output_CV_ITL_Min_Ensemble"
itlanswer = "BERT_Output_CV_ITL_Full_Answer"

def ReadAnswerLabels(path, directory, num_files=10):
    Answer = []
    for i in range(0, num_files):
        with open(os.path.join(path, directory, "Test_Data_Part_{}_Answer.tsv".format(i))) as fp:
            reader = csv.reader(fp, delimiter='\t')
            for j, row in enumerate(reader):
                if(j == 0):
                    continue
                Answer.append([row[0], row[2]])
    return Answer
            

def ReadLabels(path, directory, itl=False):
    task1_labels = []
    task2_labels = []
    task3_labels = []
    task4_labels = []
    task5_labels = []
    files = os.listdir(os.path.join(path, directory))
    for j, file in enumerate(files):
        if(file.startswith('Reqd')):
            k = int(file[-5])
            with open(os.path.join(path, directory, file)) as fp:
                reader = csv.reader(fp)
                for i, row in enumerate(reader):
                    if(itl):
                        row.insert(1, str(k-1))
                    if(i == 0):
                        continue
                    if(k == 1):
                        task1_labels.append(row)
                    elif(k == 2):
                        task2_labels.append(row)
                    elif(k == 3):
                        task3_labels.append(row)
                    elif(k == 4):
                        task4_labels.append(row)
                    elif(k == 5):
                        task5_labels.append(row)
    return task1_labels, task2_labels, task3_labels, task4_labels, task5_labels
                    
                    
label_converter = {
        0: ["NA", "O",],
        1: ["F", "T"],
        2: ["NA", "TR", ],
        3: ["F", "T"],
        4: ['FALSE', 'TRUE'],
    }                    

if(mtl):
    task1, task2, task3, task4, task5 = ReadLabels(path, mtlstep)
elif(itlanswerflag):
    task1, task2, task3, task4, task5 = ReadLabels(path, itlanswer, True)
else:    
    task1, task2, task3, task4, task5 = ReadLabels(path, itlstepcv, True)
    
Answer = ReadAnswerLabels(path, AnswerDir, 10)
task1.sort(key=lambda x: x[0])
task2.sort(key=lambda x: x[0])
task3.sort(key=lambda x: x[0])
task4.sort(key=lambda x: x[0])
task5.sort(key=lambda x: x[0])
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
ConvertLabels(task5, 5, label_converter)

compute_task2 = []
compute_task3 = []
compute_task4 = []

def DoStepEliminate(task1, task2, task3, task4):
    final_labels = [[] for i in range(len(task4))]
    originalLabelDict = dict()
    for i, row in enumerate(task1):
        if(row[3] in ['GM', 'CR']):
            final_labels[i].extend(task4[i][0:3] + ['F'])
        elif(row[3] == 'O'):
            originalLabelDict[(row[0], i)] = 1
    for i, row in enumerate(task2):
        if(len(final_labels[i]) == 0):
            compute_task2.append(row)
        if(row[3] in ['F'] and len(final_labels[i]) == 0):
            final_labels[i].extend(task4[i][0:3] + ['F'])
    for i, row in enumerate(task3):
        if(len(final_labels[i]) == 0):
            compute_task3.append(row)
        if(row[3] in ['Q', 'SE'] and len(final_labels[i]) == 0):
            final_labels[i].extend(task4[i][0:3] + ['F'])
        elif(row[3] == 'C' and (row[0], i) in originalLabelDict and len(final_labels[i]) == 0):
            final_labels[i].extend(task4[i][0:3] + ['F'])
    for i, row in enumerate(task4):
        if(len(final_labels[i]) == 0):
            compute_task4.append(row)
        if(row[3] in ['F'] and len(final_labels[i]) == 0):
            final_labels[i].extend(task4[i][0:3] + ['F'])
        elif(row[3] in ['T'] and len(final_labels[i]) == 0):
            final_labels[i].extend(task4[i][0:3] + ['T'])
    return final_labels

def DoStepEliminateTask(task1, task2, task3, task4, Answer):
    final_labels = [[] for i in range(len(Answer))]
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    totCount = 0
    for i, row in enumerate(task1):
        totCount += 1
        if(row[3] in ['GM', 'CR', 'R', 'NA']):
            final_labels[i].extend(task4[i][0:2] + [Answer[i][1]] + ['FALSE'])
            if(Answer[i][1] == 'TRUE'):
                count1 += 1
    
    for i, row in enumerate(task2):
        if(len(final_labels[i]) == 0):
            compute_task2.append(row)
        if(row[3] in ['F'] and len(final_labels[i]) == 0):
            final_labels[i].extend(task4[i][0:2] + [Answer[i][1]] + ['FALSE'])
            if(Answer[i][1] == 'TRUE'):
                count2 += 1
    
    for i, row in enumerate(task3):
        if(len(final_labels[i]) == 0):
            compute_task3.append(row)
        if(row[3] not in ['TR'] and len(final_labels[i]) == 0):
            final_labels[i].extend(task4[i][0:2] + [Answer[i][1]] + ['FALSE'])
            if(Answer[i][1] == 'TRUE'):
                count3 += 1
                
    for i, row in enumerate(task4):
        if(len(final_labels[i]) == 0):
            compute_task4.append(row)
        if(row[3] in ['F'] and len(final_labels[i]) == 0):
            final_labels[i].extend(task4[i][0:2] + [Answer[i][1]] + ['FALSE'])
            if(Answer[i][1] == 'TRUE'):
                count4 += 1
        elif(row[3] in ['T'] and len(final_labels[i]) == 0):
            final_labels[i].extend(task4[i][0:2] + [Answer[i][1]] + ['TRUE'])
            if(Answer[i][1] == 'FALSE'):
                count4 += 1
    print("CONFLICTS: \nTot Count: {}\nCount 1: {}\nCount 2: {}\nCount 3: {}\nCount 4: {}".format(
            totCount, count1, count2, count3, count4))
    return final_labels

final_labels = DoStepEliminateTask(task1, task2, task3, task4, Answer)


if(mtl):
    file_path = os.path.join(path, mtlstep, "Overall_Cross_Validation_Report.txt")
elif(itlanswerflag):
    file_path = os.path.join(path, itlanswer, "Overall_Cross_Validation_Report.txt")
else:
    file_path = os.path.join(path, itlstepcv, "Overall_Cross_Validation_Report.txt")    
file = open(file_path, 'w')
if(itlanswer):
    file.write(classification_report([row[2] for row in task5], [row[3] for row in task5]))
else:
    file.write(classification_report([row[2] for row in final_labels], [row[3] for row in final_labels]))
file.close()

if(not itlanswer):
    if(mtl):
        file_path = os.path.join(path, mtlstep, "Task_Wise_Report.txt")
    else:
        file_path = os.path.join(path, itlstepcv, "Task Wise Report.txt")
    
    file = open(file_path, 'w')
    file.write("\n\nTask 1: \n" + classification_report([row[2] for row in task1], [row[3] for row in task1]) + "\n\n")
    file.write("\n\nTask 2: \n" + classification_report([row[2] for row in compute_task2], [row[3] for row in compute_task2]) + "\n\n")
    file.write("\n\nTask 3: \n" + classification_report([row[2] for row in compute_task3], [row[3] for row in compute_task3]) + "\n\n")
    file.write("\n\nTask 4: \n" + classification_report([row[2] for row in compute_task4], [row[3] for row in compute_task4]) + "\n\n")
    file.close()