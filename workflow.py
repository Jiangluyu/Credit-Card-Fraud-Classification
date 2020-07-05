import pandas as pd
import numpy as np
from data import Data
from data_processing import DataProcess
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import itertools


def print_Kfold_scores(X_train, y_train):
    # define 5-fold cross validation
    fold = KFold(n_splits=5, shuffle=False)

    # set range of c_parameter
    c_param_range = [0.01, 0.1, 1, 10, 100]

    # init result data frame
    results = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter','Mean_recall_score'])
    results['C_parameter'] = c_param_range

    # validate the effect of different c parameters
    i = 0
    for c_param in c_param_range:
        print("-----------------------------C_param: %s------------------------" % c_param)
        print("")

        recall_accs = []

        # cross validate
        for iteration, indices in enumerate(fold.split(X_train), start=1):
            # call the logistic regression model
            lr = LogisticRegression(C = c_param, penalty= 'l1', solver='liblinear')

            lr.fit(X_train.iloc[indices[0], :], y_train.iloc[indices[0], :].values.ravel())
            y_pred_under_sampling = lr.predict(X_train.iloc[indices[1], :].values)
            recall_acc = recall_score(y_train.iloc[indices[1], :].values, y_pred_under_sampling)
            recall_accs.append(recall_acc)
            print("Iteration ", iteration, " : recall score = ", recall_acc)

        results.loc[i, 'Mean_recall_score'] = np.mean(recall_accs)
        i += 1
        print("")
        print("Mean recall score: ", np.mean(recall_accs))
        print("-----------------------------------------------------------------")

    best_c_param = results.loc[results['Mean_recall_score'].astype('float64').idxmax()]['C_parameter']

    print("Best model parameters: C parameter = ", best_c_param)

    return best_c_param


# pred with best_c_param
def get_predict_values(best_c_param, X_train, y_train, X_test):
    lr = LogisticRegression(C=best_c_param, penalty='l1', solver='liblinear')
    lr.fit(X_train, y_train.values.ravel())
    y_pred = lr.predict(X_test.values)
    return y_pred


# pred with best_c_param by probability
def get_predict_probilities(best_c_param, X_train, y_train, X_test):
    lr = LogisticRegression(C=best_c_param, penalty='l1', solver='liblinear')
    lr.fit(X_train, y_train.values.ravel())
    y_pred_proba = lr.predict_proba(X_test.values)
    return y_pred_proba


# plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def show_confusion_matrix(y_test, y_pred):
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]))

    # Plot non-normalized confusion matrix
    class_names = [0, 1]
    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=class_names, title='Confusion matrix')
    plt.show()


def show_confusion_matrix_proba(y_pred_under_sampling_proba):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    plt.figure(figsize=(10, 10))

    j = 1
    for i in thresholds:
        y_test_predictions_high_recall = y_pred_under_sampling_proba[:, 1] > i

        plt.subplot(3, 3, j)
        j += 1

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_under_sampling_test, y_test_predictions_high_recall)
        np.set_printoptions(precision=2)

        print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

        # Plot non-normalized confusion matrix
        class_names = [0, 1]
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >= %s' % i)

    plt.show()


def show_roc_curve(y_test, y_pred_score):
    fpr, tpr, thresholds = roc_curve(y_test.values.ravel(), y_pred_score)
    roc_auc = auc(fpr, tpr)
    # Plot ROC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.0])
    plt.ylim([-0.1, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == '__main__':
    # instantiate data object
    data = Data()
    data.show_sns_time_amount()
    data_process = DataProcess(data)
    data_process.standardize_amount()

    # under sampling
    X, y, X_under_sampling, y_under_sampling = data_process.under_sampling()

    # cross validation split
    X_train, X_test, y_train, y_test = data_process.cross_validation_split(X, y)
    X_under_sampling_train, X_under_sampling_test, y_under_sampling_train, y_under_sampling_test = data_process.cross_validation_split(X_under_sampling, y_under_sampling)

    # ------------------------------------------split line-----------------------------------------
    # based on under sampling
    # compute best C parameter
    print("---------------------------split line----------------------------")
    best_c_param = print_Kfold_scores(X_under_sampling_train, y_under_sampling_train)

    # compute and plot confusion matrix
    y_under_sampling_pred = get_predict_values(best_c_param, X_under_sampling_train, y_under_sampling_train, X_under_sampling_test)
    show_confusion_matrix(y_under_sampling_test, y_under_sampling_pred)

    y_pred = get_predict_values(best_c_param, X_under_sampling_train, y_under_sampling_train, X_test)
    show_confusion_matrix(y_test, y_pred)

    lr = LogisticRegression(C=best_c_param, penalty='l1', solver='liblinear')
    y_pred_under_sampling_score = lr.fit(X_under_sampling_train, y_under_sampling_train.values.ravel()).decision_function(
        X_under_sampling_test.values)
    show_roc_curve(y_under_sampling_test, y_pred_under_sampling_score)

    # ------------------------------------------split line-----------------------------------------
    # based on raw dataset (for comparison)
    # compute best C parameter
    print("---------------------------split line----------------------------")
    best_c_param_raw = print_Kfold_scores(X_train, y_train)

    # compute and plot confusion matrix
    y_pred_raw = get_predict_values(best_c_param_raw, X_train, y_train, X_test)
    show_confusion_matrix(y_test, y_pred_raw)

    lr = LogisticRegression(C=best_c_param, penalty='l1', solver='liblinear')
    y_pred_score = lr.fit(X_train, y_train.values.ravel()).decision_function(X_test.values)
    show_roc_curve(y_test, y_pred_score)
    # ------------------------------------------split line-----------------------------------------
    # based on probability
    print("---------------------------split line----------------------------")
    y_pred_under_sampling_proba = get_predict_probilities(best_c_param, X_under_sampling_train, y_under_sampling_train, X_under_sampling_test)
    show_confusion_matrix_proba(y_pred_under_sampling_proba)







