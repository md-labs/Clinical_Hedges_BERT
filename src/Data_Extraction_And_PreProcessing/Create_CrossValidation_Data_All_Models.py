"""
Used to create Cross Validation for any number of folds for any of the four models described in this approach. Additionally, it balances the data for each task and does some preliminary preprocessing on the text
Input:
MTL_Task_Labels_Marshall.csv OR MTL_Task_Labels_Del_Fiol.csv, Final_Retrieved_Clinical_Hedges.csv
Output:
CV_Data_ITL, CV_Data_Ensemble, CV_Data_ITL_Ensemble, CV_Data_ITL_Cascade
"""

import os
import csv
from collections import defaultdict
import random
import shutil
import nltk

random.seed(42)

path_to_data = os.path.abspath("../../Dataset/")
unfound = []


def ShuffleData(data):
    index = []
    for i in range(len(data)):
        index.append(i)
    req_index = random.sample(index, len(data))
    shuffled_data = []
    for index in req_index:
        shuffled_data.append(data[index])
    return shuffled_data


def ReadData(file_name):
    RetList = []
    with open(os.path.join(path_to_data, file_name), encoding='utf8') as fp:
        reader = csv.reader(fp)
        for row in reader:
            RetList.append(row)
    return RetList


def CreateBERTTypeData(DocList, MTLDict):
    task1 = []
    task2 = []
    task3 = []
    task4 = []
    Answer = []
    count = 0
    for row in DocList:
        if(len(MTLDict[row[0]]) != 0):
            label_list = MTLDict[row[0]].pop()
        else:
            continue
        if((row[1] == '' and row[2] == '') or row[2] == ''): # Use for Marshall
        # if (row[1] == '' and row[2] == ''):
            count += 1
            unfound.append(row[0])
            continue
        # If PT Terms to be included along with Title and Abstract
        task1.append([row[0], ' '.join((row[3] + row[1] + ' ' + row[2]).split('\n')), label_list[0], 0])
        task2.append([row[0], ' '.join((row[3] + row[1] + ' ' + row[2]).split('\n')), label_list[1], 1])
        task3.append([row[0], ' '.join((row[3] + row[1] + ' ' + row[2]).split('\n')), label_list[2], 2])
        task4.append([row[0], ' '.join((row[3] + row[1] + ' ' + row[2]).split('\n')), label_list[3], 3])
        # If only Title and Abstract are to be included
        #task1.append([row[0], ' '.join((row[1] + ' ' + row[2]).split('\n')), label_list[0], 0])
        #task2.append([row[0], ' '.join((row[1] + ' ' + row[2]).split('\n')), label_list[1], 1])
        #task3.append([row[0], ' '.join((row[1] + ' ' + row[2]).split('\n')), label_list[2], 2])
        #task4.append([row[0], ' '.join((row[1] + ' ' + row[2]).split('\n')), label_list[3], 3])
        Answer.append([row[0], ' '.join((row[1] + ' ' + row[2]).split('\n')), label_list[4], 4])
    return task1, task2, task3, task4, Answer


def CountLabels(data):
    count_labels = dict()
    for row in data:
        if(row[2] not in count_labels):
            count_labels[row[2]] = 1
        else:
            count_labels[row[2]] += 1
    return count_labels


def UnderSampleData(data, reqDistDict):
    count_added = dict()
    undersampleData = []
    for row in data:
        if(row[2] not in count_added):
            if(reqDistDict[row[2]] == 0):
                continue
            count_added[row[2]] = 1
            undersampleData.append(row)
        elif(count_added[row[2]] < reqDistDict[row[2]]):
                count_added[row[2]] += 1
                undersampleData.append(row)
    return undersampleData


def OversampleData(data):
    count_added = dict()
    count_labels = CountLabels(data)
    maxCount = 0
    minCount = 999999999999
    for key, value in count_labels.items():
        if key != 'NA':
            maxCount = max(value, maxCount)
            minCount = min(value, minCount)
    binary = False
    count_reqd = dict()
    if('F' in count_labels):
        binary = True
        count_reqd['F'] = int(count_labels['F'] * count_labels['T'] / (count_labels['F'] + count_labels['NA']))  if 'NA' in count_labels else count_labels['T']
        count_reqd['NA'] = int(count_labels['NA'] * count_labels['T'] / (count_labels['F'] + count_labels['NA'])) if 'NA' in count_labels else 0
        count_reqd['T'] = int(count_labels['T'])
    else:
        for label in count_labels:
            # count_reqd[label] = int((maxCount + minCount) / 2)
            count_reqd[label] = minCount
    oversampleData = []
    while True:
        for row in data:
            if(row[2] not in count_added):
                if(count_reqd[row[2]] == 0):
                    continue
                count_added[row[2]] = 1
                appender = list(row)
                appender[2] = 'F' if appender[2] == 'NA' and binary else appender[2]
                oversampleData.append(appender)
            elif(count_added[row[2]] < count_reqd[row[2]]):
                count_added[row[2]] += 1
                appender = list(row)
                appender[2] = 'F' if appender[2] == 'NA' and binary else appender[2]
                oversampleData.append(appender)
        oversample = False
        for key in count_added:
            if(count_added[key] < count_reqd[key]):
                oversample = True
        if(not oversample):
            break
    return oversampleData


def checkBatchTask(data, batch_size=16):
    for i in range(0, len(data), batch_size):
        batch_task = [row[3] for row in data[i:i+batch_size]]
        label = batch_task[0]
        for task in batch_task:
            if task != label:
                return False
    return True


def SplitData(data, test_per, batch_size):
    test_size = int(test_per * len(data))
    test_size += (len(data) - test_size) % batch_size
    train_data = data[:len(data)-test_size]
    test_data = data[len(data)-test_size:]
    return train_data, test_data


def ConstructSet(task1, task2, task3, task4, batch_size):
    resultData = []
    dataLen = max(len(task1), max(len(task2), max(len(task3), len(task4))))
    while(dataLen % batch_size != 0):
        dataLen -= 1
    for i in range(0, dataLen, batch_size):
        if(i + batch_size < len(task1)):
            resultData.extend(task1[i: i+batch_size])
        if(i + batch_size < len(task2)):
            resultData.extend(task2[i: i+batch_size])
        if(i + batch_size < len(task3)):
            resultData.extend(task3[i: i+batch_size])
        if(i + batch_size < len(task4)):
            resultData.extend(task4[i: i+batch_size])
    print(checkBatchTask(resultData, batch_size))
    return resultData


def ConvertNAToF(data):
    resultData = []
    for row in data:
        row[2] = 'F' if row[2] == 'NA' else row[2]
        resultData.append(row)
    return resultData


def filterLabelsTask2(task1, task2):
    filter1_labels = ['NA',]
    result = []
    task1Dict = dict()
    for row in task1:
        if(row[2] in filter1_labels):
            task1Dict[row[0]] = 1
    for row in task2:
        if(row[0] not in task1Dict):
            result.append(row)
    return result


def filterLabelsTask3(task1, task2, task3):
    task3 = filterLabelsTask2(task1, task3)
    filter2_labels = ['F']
    result = []
    task2Dict = dict()
    for row in task2:
        if(row[2] in filter2_labels):
            task2Dict[row[0]] = 1
    for row in task3:
        if(row[0] not in task2Dict):
            result.append(row)
    return result


def filterLabelsTask4(task1, task2, task3, task4):
    task4 = filterLabelsTask3(task1, task2, task4)
    filter3_labels = ['NA']
    result = []
    task3Dict = dict()
    task1Dict = dict()
    for row in task1:
        if(row[2] == 'O'):
            task1Dict[row[0]] = 1
    for row in task3:
        if(row[2] in filter3_labels or (row[2] == 'C' and row[0] in task1Dict)):
            task3Dict[row[0]] = 1
    for row in task4:
        if(row[0] not in task3Dict):
            result.append(row)
    return result


def changeLabels(task, task_num):
    result = []
    for row in task:
        if(task_num == 1):
            if(row[2] == 'O'):
                result.append(row)
            else:
                result.append(row[0:2] + ['NA'] + [row[3]])
        elif(task_num == 3):
            if(row[2] == 'TR'):
                result.append(row)
            else:
                result.append(row[0:2] + ['NA'] + [row[3]])
    return result


def WriteData(file_name, data, path=path_to_data):
    with open(os.path.join(path, file_name), 'w', newline='', encoding='utf-8') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerow(["ID", "Text", "Label", "Task_Number"])
        for row in data:
            writer.writerow(row)


def CreateCrossValidationITL(task1, task2, task3, task4, Answer, data_dir, batch_size, 
                             proportion=None, num_folds=1, toFilter=False, itl=False):
    if(os.path.exists(os.path.join(path_to_data, data_dir))):
        shutil.rmtree(os.path.join(path_to_data, data_dir))
    os.mkdir(os.path.join(path_to_data, data_dir))
    task1 = changeLabels(task1, 1)
    task3 = changeLabels(task3, 3)
    if(num_folds == 1):
        test_size = int(len(task4) * 0.30) + 1
    else:
        test_size = int(len(task4) * 1 / num_folds) + 1
    count_folds = 0
    for test_index in range(0, len(task4), test_size):
        if(count_folds >= num_folds):
            break
        
        train_task1 = task1[0:test_index] + task1[test_index + test_size:]
        train_task2 = task2[0:test_index] + task2[test_index + test_size:]
        train_task3 = task3[0:test_index] + task3[test_index + test_size:]
        train_task4 = task4[0:test_index] + task4[test_index + test_size:]
        train_Answer = Answer[0:test_index] + Answer[test_index + test_size:]
        
        test_task1 = task1[test_index: test_index + test_size]
        test_task2 = task2[test_index: test_index + test_size]
        test_task3 = task3[test_index: test_index + test_size]
        test_task4 = task4[test_index: test_index + test_size]
        test_Answer = Answer[test_index: test_index + test_size]
        
        train_task1 = ShuffleData(train_task1)
        train_task2 = ShuffleData(train_task2)
        train_task3 = ShuffleData(train_task3)
        train_task4 = ShuffleData(train_task4)
        train_Answer = ShuffleData(train_Answer)
        
        if(toFilter):
            train_task2 = filterLabelsTask2(task1, train_task2)
            train_task3 = filterLabelsTask3(task1, task2, train_task3)
            train_task4 = filterLabelsTask4(task1, task2, task3, train_task4)
        
        train_task1 = OversampleData(train_task1)
        train_task2 = OversampleData(train_task2)
        train_task3 = OversampleData(train_task3)
        train_task4 = OversampleData(train_task4)
        train_Answer = OversampleData(train_Answer)
        
        test_task4 = ConvertNAToF(test_task4)
        test_task2 = ConvertNAToF(test_task2)
        
        print("Fold: ", count_folds + 1)
        print(CountLabels(train_Answer))

        if not itl:
            WriteData("Temp_Train_Data_Part_{}_Task_1.tsv".format(int(test_index / test_size)), train_task1[:batch_size], os.path.join(path_to_data, data_dir))
            WriteData("Train_Data_Part_{}_Task_1.tsv".format(int(test_index / test_size)), train_task1, os.path.join(path_to_data, data_dir))
            WriteData("Dev_Data_Part_{}_Task_1.tsv".format(int(test_index / test_size)), train_task1[:batch_size], os.path.join(path_to_data, data_dir))
            WriteData("Test_Data_Part_{}_Task_1.tsv".format(int(test_index / test_size)), test_task1, os.path.join(path_to_data, data_dir))

            WriteData("Temp_Train_Data_Part_{}_Task_2.tsv".format(int(test_index / test_size)), train_task2[:batch_size], os.path.join(path_to_data, data_dir))
            WriteData("Train_Data_Part_{}_Task_2.tsv".format(int(test_index / test_size)), train_task2, os.path.join(path_to_data, data_dir))
            WriteData("Dev_Data_Part_{}_Task_2.tsv".format(int(test_index / test_size)), train_task2[:batch_size], os.path.join(path_to_data, data_dir))
            WriteData("Test_Data_Part_{}_Task_2.tsv".format(int(test_index / test_size)), test_task2, os.path.join(path_to_data, data_dir))

            WriteData("Temp_Train_Data_Part_{}_Task_3.tsv".format(int(test_index / test_size)), train_task3[:batch_size], os.path.join(path_to_data, data_dir))
            WriteData("Train_Data_Part_{}_Task_3.tsv".format(int(test_index / test_size)), train_task3, os.path.join(path_to_data, data_dir))
            WriteData("Dev_Data_Part_{}_Task_3.tsv".format(int(test_index / test_size)), train_task3[:batch_size], os.path.join(path_to_data, data_dir))
            WriteData("Test_Data_Part_{}_Task_3.tsv".format(int(test_index / test_size)), test_task3, os.path.join(path_to_data, data_dir))

            WriteData("Temp_Train_Data_Part_{}_Task_4.tsv".format(int(test_index / test_size)), train_task4[:batch_size], os.path.join(path_to_data, data_dir))
            WriteData("Train_Data_Part_{}_Task_4.tsv".format(int(test_index / test_size)), train_task4, os.path.join(path_to_data, data_dir))
            WriteData("Dev_Data_Part_{}_Task_4.tsv".format(int(test_index / test_size)), train_task4[:batch_size], os.path.join(path_to_data, data_dir))
            WriteData("Test_Data_Part_{}_Task_4.tsv".format(int(test_index / test_size)), test_task4, os.path.join(path_to_data, data_dir))

        WriteData("Temp_Train_Data_Part_{}_Answer.tsv".format(int(test_index / test_size)), train_Answer[:batch_size], os.path.join(path_to_data, data_dir))
        WriteData("Train_Data_Part_{}_Answer.tsv".format(int(test_index / test_size)), train_Answer, os.path.join(path_to_data, data_dir))
        WriteData("Dev_Data_Part_{}_Answer.tsv".format(int(test_index / test_size)), train_Answer[:batch_size], os.path.join(path_to_data, data_dir))
        WriteData("Test_Data_Part_{}_Answer.tsv".format(int(test_index / test_size)), test_Answer, os.path.join(path_to_data, data_dir))
        
        count_folds += 1


def CreateCrossValidationMTL(task1, task2, task3, task4, Answer, data_dir, batch_size, num_folds=1):
    if(os.path.exists(os.path.join(path_to_data, data_dir))):
        shutil.rmtree(os.path.join(path_to_data, data_dir))
    os.mkdir(os.path.join(path_to_data, data_dir))
    task1 = changeLabels(task1, 1)
    task3 = changeLabels(task3, 3)
    if(num_folds == 1):
        test_size = int(len(task4) * 0.30) + 1
    else:
        test_size = int(len(task4) * 1 / num_folds) + 1
    count_folds = 0
    for test_index in range(0, len(task4), test_size):
        if(count_folds >= num_folds):
            break
        train_task1 = task1[0:test_index] + task1[test_index + test_size:]
        train_task2 = task2[0:test_index] + task2[test_index + test_size:]
        train_task3 = task3[0:test_index] + task3[test_index + test_size:]
        train_task4 = task4[0:test_index] + task4[test_index + test_size:]
        
        test_task1 = task1[test_index: test_index + test_size]
        test_task2 = task2[test_index: test_index + test_size]
        test_task3 = task3[test_index: test_index + test_size]
        test_task4 = task4[test_index: test_index + test_size]
        test_Answer = Answer[test_index: test_index + test_size]
        
        train_task1 = ShuffleData(train_task1)
        train_task2 = ShuffleData(train_task2)
        train_task3 = ShuffleData(train_task3)
        train_task4 = ShuffleData(train_task4)
        
        train_task1 = OversampleData(train_task1)
        train_task2 = OversampleData(train_task2)
        train_task3 = OversampleData(train_task3)
        train_task4 = OversampleData(train_task4)
        
        test_task2 = ConvertNAToF(test_task2)
        test_task4 = ConvertNAToF(test_task4)
        full_train_data = ConstructSet(train_task1, train_task2, train_task3, train_task4, batch_size)
        print("Fold: ", count_folds + 1)
        print("Test: \n", CountLabels(test_task4))
        print("Train MTL Batch Wise Labels= ", checkBatchTask(full_train_data))
        print()
        
        WriteData("Temp_Train_Data_Part_{}.tsv".format(int(test_index / test_size)), full_train_data[:batch_size*4], os.path.join(path_to_data, data_dir))
        WriteData("Train_Data_Part_{}.tsv".format(int(test_index / test_size)), full_train_data, os.path.join(path_to_data, data_dir))
        WriteData("Dev_Data_Part_{}.tsv".format(int(test_index / test_size)), full_train_data[:batch_size], os.path.join(path_to_data, data_dir))
        WriteData("Test_Data_Part_{}_Task_1.tsv".format(int(test_index / test_size)), test_task1, os.path.join(path_to_data, data_dir))
        WriteData("Test_Data_Part_{}_Task_2.tsv".format(int(test_index / test_size)), test_task2, os.path.join(path_to_data, data_dir))
        WriteData("Test_Data_Part_{}_Task_3.tsv".format(int(test_index / test_size)), test_task3, os.path.join(path_to_data, data_dir))
        WriteData("Test_Data_Part_{}_Task_4.tsv".format(int(test_index / test_size)), test_task4, os.path.join(path_to_data, data_dir))
        WriteData("Test_Data_Part_{}_Answer.tsv".format(int(test_index / test_size)), test_Answer, os.path.join(path_to_data, data_dir))
        
        count_folds += 1


def main():
    MTL_Labels = ReadData("MTL_Task_Labels_Del_Fiol.csv")
    # MTL_Labels = ReadData("MTL_Task_Labels_Marshall.csv")
    RetCHData = ReadData("Final_Retrieved_Clinical_Hedges.csv")
    
    RetCHData = ShuffleData(RetCHData)
    
    MTL_LabelDict = defaultdict(list)
    for row in MTL_Labels:
        MTL_LabelDict[row[0]].append(row[1:])

    task1, task2, task3, task4, Answer = CreateBERTTypeData(RetCHData, MTL_LabelDict)
    
    batch_size = 16
    
    CreateCrossValidationITL(task1, task2, task3, task4, Answer, 
                             "CV_Data_ITL", batch_size, num_folds=10, itl=True)
    CreateCrossValidationITL(task1, task2, task3, task4, Answer,
                             "CV_Data_Ensemble", batch_size, num_folds=10, itl=True)
    CreateCrossValidationITL(task1, task2, task3, task4, Answer, "CV_Data_ITL_Ensemble",
                            batch_size, num_folds=10)
    CreateCrossValidationITL(task1, task2, task3, task4, Answer, "CV_Data_ITL_Cascade",
                            batch_size, num_folds=10, toFilter=True)
    # CreateCrossValidationMTL(task1, task2, task3, task4, Answer, "CV_Data_MTL",
    #                         batch_size, num_folds=10)

    length = []
    for row in RetCHData:
        length.append(len(nltk.word_tokenize(' '.join((row[1] + ' ' + row[2]).split('\n')))))
    print(RetCHData[0])
    return length


if __name__ == '__main__':
    length = main()
    # Used to calculate optimium sequence length for data
    avg = sum(length) / len(length)
    maximum = max(length)
    length = list(sorted(length))
    for i, l in enumerate(length):
        if l > 384:
            print("384th Percentile: ", (i / len(length)) * 100)
            break
    percentile = 0.95 * len(length)
    print("95th Percentile: ", length[int(percentile)])
    percent_list = length[:int(percentile)]
    print("95th Percentile Avg: ", sum(percent_list) / int(percentile))
    print("Average: ", avg)
    print("Maximum: ", maximum)
