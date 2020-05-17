"""
Contains sklearns classification report modified to print precision, recall and f-score with more floating point precision
(increased floating point precision to 4 decimal points)
"""

import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score


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

    microf1 = f1_score(y_true, y_pred, average='micro')
    micropre = precision_score(y_true, y_pred, average='micro')
    microrec = recall_score(y_true, y_pred, average='micro')
    
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
