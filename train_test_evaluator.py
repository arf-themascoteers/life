from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


def average_accuracy(y_true, y_pred):
    ca = []
    for c in np.unique(y_true):
        y_c = y_true[np.nonzero(y_true == c)]
        y_c_p = y_pred[np.nonzero(y_true == c)]
        acurracy = accuracy_score(y_c, y_c_p)
        ca.append(acurracy)
    ca = np.array(ca)
    aa = ca.mean()
    return aa


def evaluate_train_test_pair(train_x, test_x, train_y, test_y, classification=True):
    evaluator_algorithm = get_metric_evaluator(classification)
    evaluator_algorithm.fit(train_x, train_y)
    y_pred = evaluator_algorithm.predict(test_x)
    return calculate_metrics(test_y, y_pred, classification)


def evaluate_split(train_x, test_x, train_y, test_y, transform=None, classification=True):
    if transform is not None:
        train_x = transform.transform(train_x)
        test_x = transform.transform(test_x)
    return evaluate_train_test_pair(train_x, test_x, train_y, test_y, classification)


def calculate_metrics(y_test, y_pred, classification=True):
    if classification:
        oa = accuracy_score(y_test, y_pred)
        aa = average_accuracy(y_test, y_pred)
        k = cohen_kappa_score(y_test, y_pred)
        return oa, aa, k

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return r2, rmse, 0

def get_metric_evaluator(classification=True):
    if classification:
        return SVC(C=1e5, kernel='rbf', gamma=1.)
    else:
        return SVR(C=1e5, kernel='rbf', gamma=1.)