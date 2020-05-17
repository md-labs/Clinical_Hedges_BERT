"""
Extract Clinical Hedges articles from given Excel file according to conditions set by Marshall / Del Fiol
Input:
Clinical Hedges Data - Medline.xls
Output:
All_Clincial_Hedges_Articles.csv, MTL_Task_Labels_Marshall.csv
"""

import os
import xlrd
import csv
import re

path_to_data = "C:/Clinical_Hedges/Dataset/"
wb = xlrd.open_workbook(os.path.join(path_to_data, 'Clinical Hedges Data - Medline.xls')) 
sheet = wb.sheet_by_index(0)

column1 = {'GM': 'GM',
           'CR': 'CR',
           '': 'NA',
           'O': 'O',
           'R': 'R',
        }

column2 = {'1': 'T',
           '0': 'F',
           '': 'NA',
        }

column3 = {'': 'NA',
           'C': 'C',
           'SE': 'SE',
           'CPG': 'CPG',
           'D': 'D',
           'E': 'E',
           'Ec': 'EC',
           'P': 'P',
           'Qual': 'Q',
           'Tr': 'TR',
        }

column4 = {'1': 'T',
           '0': 'F',
           '': 'NA',
        }

all_articles = []
task_labels = []
cSound = []
cNSound = []
full_data = []
count_tr = 0
count = 0
Tcount = 0
for i in range(sheet.nrows):
    if(i == 0):
        continue
    all_articles.append(str(int(sheet.cell_value(i, 0))))
    if(re.sub(' +', '', sheet.cell_value(i, 1)) == 'O'):
        count_tr += 1
    if(re.sub(' +', '', sheet.cell_value(i, 3)) == 'Tr'and sheet.cell_value(i, 4) == 1 and
       re.sub(' +', '', sheet.cell_value(i, 1)) == 'O'):
            cSound.append(str(int(sheet.cell_value(i, 0))))
            curList = []
            curList.append(str(int(sheet.cell_value(i, 0))))
            col1 = re.sub(' +', '', str(sheet.cell_value(i, 1)))
            curList.append(column1[col1])
            col2 = re.sub(' +', '', str(sheet.cell_value(i, 2)))
            curList.append(column2[col2])
            col3 = re.sub(' +', '', str(sheet.cell_value(i, 3)))
            curList.append(column3[col3])
            col4 = re.sub(' +', '', str(sheet.cell_value(i, 4)))
            curList.append(column4[col4])
            curList.append("TRUE")
            task_labels.append(curList)
    else:
        cNSound.append(str(int(sheet.cell_value(i, 0))))
        curList = []
        curList.append(str(int(sheet.cell_value(i, 0))))
        col1 = re.sub(' +', '', str(sheet.cell_value(i, 1)))
        curList.append(column1[col1])
        col2 = re.sub(' +', '', str(sheet.cell_value(i, 2)))
        curList.append(column2[col2])
        col3 = re.sub(' +', '', str(sheet.cell_value(i, 3)))
        curList.append(column3[col3])
        col4 = re.sub(' +', '', str(sheet.cell_value(i, 4)))
        curList.append(column4[col4])
        curList.append("FALSE")
        task_labels.append(curList)
    
with open(os.path.join(path_to_data, "All_Clincial_Hedges_Articles.csv"), 'w', newline='') as fp:
    writer = csv.writer(fp)
    for row in all_articles:
        writer.writerow([row])

with open(os.path.join(path_to_data, "MTL_Task_Labels_Marshall.csv"), 'w', newline='') as fp:
    writer = csv.writer(fp)
    for row in task_labels:
        writer.writerow(row)
