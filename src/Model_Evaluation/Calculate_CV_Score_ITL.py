import os
import csv
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

def classification_report(y_true, y_pred, labels=None, target_names=None,
                          sample_weight=None):
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    last_line_heading = 'avg / total'

    if target_names is None:
        width = len(last_line_heading)
        target_names = ['%s' % l for l in labels]
    else:
        width = max(len(cn) for cn in target_names)
        width = max(width, len(last_line_heading))

    headers = ["precision", "recall", "f1-score", "support"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None,
                                                  sample_weight=sample_weight)
    #print(s)
    #print(np.average(p, weights=s))
    microf1 = f1_score(y_true, y_pred, average='micro')
    micropre = precision_score(y_true, y_pred, average='micro')
    microrec = recall_score(y_true, y_pred, average='micro')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    for i, label in enumerate(labels):
        values = [target_names[i]]
        for v in (p[i], r[i], f1[i]):
            values += ["{0:0.4f}".format(v)]
        values += ["{0}".format(s[i])]
        report += fmt % tuple(values)

    report += '\n'
    
    # compute averages
    values = [last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["{0:0.4f}".format(v)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)
    
    values = ["micro avg"]
    values += ["{0:0.4f}".format(micropre)]
    values += ["{0:0.4f}".format(microrec)]
    values += ["{0:0.4f}".format(microf1)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)
    
    return report


path = "C:/Clinical_Hedges/Dataset/Classification_Reports/"
balanced = "BERT_Output_CV_Balanced"
balanced_PP_50 = "BERT_Output_CV_Balanced_100_50"
balanced_PP_25 = "BERT_Output_CV_Balanced_100_25"
balanced_PP_12 = "BERT_Output_CV_Balanced_200_12"
balanced_PP_1 = "BERT_Output_CV_Balanced_200_1"
utbalanced = "BERT_Output_CV_Balanced_UT"
normal = "BERT_Output_CV"
utnormal = "BERT_Output_CV_UT"
itl = "BERT_Output_ITL"
cvbase = "BERT_Output_CV_Base"
cvbb = "BERT_Output_CV_BB"
cvsb = "BERT_Output_CV_SB"
cvut = "BERT_Output_CV_UT"
cvmtl = "BERT_Output_CV_MTL"
cvensemble = "BERT_Output_Ensemble_CV"
ensemble = "BERT_Output_Ensemble"
itlcv = "BERT_Output_CV_ITL"
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
    Cfile.close()

#CVScore(path, balanced)
#CVScore(path, balanced_PP_50)
#CVScore(path, balanced_PP_25)
#CVScore(path, balanced_PP_12)
#CVScore(path, balanced_PP_1)
#CVScore(path, utbalanced)
#CVScore(path, normal)
#CVScore(path, utnormal)
#CVScore(path, itl)
#CVScore(path, cvbase)
#CVScore(path, cvsb)
#CVScore(path, cvbb)
#CVScore(path, cvut)
#CVScore(path, cvmtl)
#CVScore(path, cvensemble)
#CVScore(path, ensemble)
#CVScore(path, itlcv)
CVScore(path, cvensemble)

"""
precision, recall, fscore = CVScore(path, balanced, 2)

print("\n\nPrecision: {}\nRecall: {}\nF-Score: {}".format(precision, recall, fscore))

precision, recall, fscore = CVScore(path, normal, 2)

print("\n\nPrecision: {}\nRecall: {}\nF-Score: {}".format(precision, recall, fscore))
"""