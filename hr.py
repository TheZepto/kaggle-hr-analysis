import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import factorize

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, precision_recall_curve

# Load the data into arrays
# Returns:
#   arrInput - input array with each row an example
#   arrTarget - target array of whether the example left


def load_hr_data():
    #Load the csv data
    dsHR = read_csv('Datasets/HR.csv')

    # Retreive the column names of the dataset
    col_names = dsHR.columns.values
    # Initialise blank arrays to load data into
    rowTarget = np.array([])
    rowInput = np.array([])
    # Read each column data into either the input or target array
    for column in col_names:
        # Build target array
        if column == 'left':
            rowTarget = dsHR[column]
        # Build the input array for the sales column of strings
        elif column == 'sales':
            encSales = factorize(dsHR[column]) # Returns the indices for each unique string label 
            rowInput = np.append(rowInput, encSales[0])
        # Build the input array for the salary column of strings
        elif column == 'salary':
            encSalary = factorize(dsHR[column])
            rowInput = np.append(rowInput, encSalary[0])
        # Build the input array for the other columns that contain numbers
        else:
            rowInput = np.append(rowInput, dsHR[column])

    # Need to reshape the arrays to be compatible with scikit-learn
    # The arrInput need to be transposed to get a shape n_samples, n_features
    arrInput = rowInput.reshape(len(col_names)-1, -1).transpose()
    arrTarget = rowTarget.values.reshape(-1,)

    return (arrInput, arrTarget);

# Train the inputted classifier using a stratified K-fold approach
# (split into 5) and evaluates the performance using cross validation
# Inputs:
#   - clf is the classifier defined from scikit-learn
#   - arrInput is a numpy array of shape n_samples, n_features
#   - arrTarget is a numpy array of shape n_samples
#   - show_report is a boolean value on whether to print the report to screen
# Returns:
#   - dict with trained_clf and the performance values

def train_classifier(clf, arrInput, arrTarget, show_report=False):
    # Generate K-fold splitting 
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    # Initialise performance metrics
    accuracy = []
    precision = []
    recall = []
    F1 = []
    # Train the classifier for each K-fold split
    for train_index, test_index in skf.split(arrInput, arrTarget):
        X_train, X_test = arrInput[train_index], arrInput[test_index]
        y_train, y_test = arrTarget[train_index], arrTarget[test_index]

        # Train the classifier on the train data
        clf.fit(X_train, y_train)

        # Generate the cross val predictions on the test data
        y_pred = clf.predict(X_test)

        # Calculate performance metrics
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        F1.append(f1_score(y_test, y_pred))

    # Compute the combined mean and 2 sigma error for the K-fold iterations
    accuracy = [np.mean(accuracy), 2*np.std(accuracy)]
    precision = [np.mean(precision), 2*np.std(precision)]
    recall = [np.mean(recall), 2*np.std(recall)]
    F1 = [np.mean(F1), 2*np.std(F1)]

    # Print the performance metrics of the classifier (opt)
    if show_report == True:
        print("**** Training Report from KFold cross validation ****")
        print("Accuracy: {:.4f} +/- {:.4f}.".format(accuracy[0], accuracy[1]) )
        print("Precision: {:.4f} +/- {:.4f}.".format(precision[0], precision[1]) )
        print("Recall: {:.4f} +/- {:.4f}.".format(recall[0], recall[1]) )
        print("F1 score: {:.4f} +/- {:.4f}.".format(F1[0], F1[1]) )
        print("")

    return {'trained_clf': clf,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'F1': F1}

# Search to optimise a classifier over a parameter grid and plot
# an optimisation graph wrt training time.
# Inputs:
#   - clf is the classifier defined from scikit-learn
#   - parameter_grid is a dict of the parameters and ranges to be searched over
#   - arrInput is a numpy array of shape n_samples, n_features
#   - arrTarget is a numpy array of shape n_samples
def optimise_classifier_f1(clf, parameter_grid, arrInput, arrTarget):
    # Generate cross validation set
    cv_splits = StratifiedKFold(n_splits=5, shuffle=True)

    # Perform grid search over paramter grid
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=parameter_grid,
        cv=cv_splits,
        scoring='f1'
        )
    grid_search.fit(arrInput, arrTarget)

    # Performance metrics from grid search
    fit_time = grid_search.cv_results_['mean_fit_time']
    fit_time_err = grid_search.cv_results_['std_fit_time']
    test_score = grid_search.cv_results_['mean_test_score']
    test_score_err = grid_search.cv_results_['std_test_score']
    rank_test_score = grid_search.cv_results_['rank_test_score']
    params = grid_search.cv_results_['params']

    # Print each point's rank and parameters
    print()
    print("Rank and parameters for: {}".format(type(clf).__name__))
    for i,j in zip(rank_test_score,params):
        print("Rank {0:2d}  Parameters {1:}".format(i,j))

    # Plot the performance graph with each point's rank
    fig, ax = plt.subplots(1,1)
    ax.errorbar(fit_time, test_score, fit_time_err, test_score_err, 'b.')
    ax.set_xlabel('Training time (s)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Evaluating {} Performance'.format(type(clf).__name__))
    for x,y,rank in zip(fit_time, test_score, rank_test_score):
        ax.annotate(rank, xy=(x,y), textcoords='data')
    plt.show()

# Plot the confusion matrix
# Updated function from the scikit-learn confusion matrix example
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure() #Generate a new plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Plot the ROC curve
def plot_ROC_curve(y_test,y_pred):
    #Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_test,y_pred)
    # Plot the figure
    plt.figure()
    plt.plot(fpr, tpr, 'bx-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteritic curve')
    plt.grid()
    plt.show()