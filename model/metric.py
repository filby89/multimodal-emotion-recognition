import torch
import sklearn.metrics
import numpy as np
import warnings

def accuracy(output, target):
    # with torch.no_grad():
    #     pred = torch.argmax(output, dim=1)
    #     assert pred.shape[0] == len(target)
    #     correct = 0
    #     correct += torch.sum(pred == target).item()
    # return correct / len(target)
    # print(output)
    # print(target)
    output = np.argmax(output, axis=1)
    return sklearn.metrics.accuracy_score(target, output)

def balanced_accuracy(output, target):
    # with torch.no_grad():
    #     pred = torch.argmax(output, dim=1)
    #     assert pred.shape[0] == len(target)
    #     correct = 0
    #     correct += torch.sum(pred == target).item()
    # return correct / len(target)
    # print(output)
    # print(target)
    output = np.argmax(output, axis=1)
    return sklearn.metrics.balanced_accuracy_score(target, output)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def average_precision(output, target):
    return sklearn.metrics.average_precision_score(target, output, average=None)

def f1_score(output, target):
    return sklearn.metrics.f1_score(target, output, average=None)

def multilabel_confusion_matrix(output, target):
    # with warnings.catch_warnings():
        # warnings.simplefilter("ignore")
    return sklearn.metrics.multilabel_confusion_matrix(target, output)


def roc_auc(output, target, average=None):
    # print(np.sum(target.cpu().detach().numpy(),axis=1),np.sum(target.cpu().detach().numpy(),axis=0))
    # print(output.size())
    return sklearn.metrics.roc_auc_score(target, output, average=average)


def mean_squared_error(output, target):
    return sklearn.metrics.mean_squared_error(target, output, multioutput='raw_values')

def r2(output, target):
    return sklearn.metrics.r2_score(target, output, multioutput='raw_values')

def ERS(mR2, mAP, mRA):
    return 1/2 * (mR2 + 1/2 * (mAP + mRA))