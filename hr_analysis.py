# My analysis methods
import hr
# Import the classifiers to use from scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# Extra utilites from scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load in the HR data
(arrInput, arrTarget) = hr.load_hr_data()
# Split data into training and testing sets
TrainInput, TestInput, TrainTarget, TestTarget = train_test_split(
    arrInput, arrTarget, test_size=0.2)
# Use standard scaler to scale features -1 to +1 mean and std
scaler = StandardScaler()
ScaledTrainInput = scaler.fit_transform(TrainInput)
ScaledTestInput = scaler.transform(TestInput)

# Part 1: Finding the best learning algorithms for the data
if input("Perform search of learning algorithms? (y/n): ") == 'y':
    # Define the classifiers to train on the data
    classifier_algorithms = {
        'Logistic Regression': LogisticRegression(),
        'Linear SVM': SVC(kernel='linear'),
        'Poly SVM': SVC(kernel='poly'),
        'RBF SVM': SVC(kernel='rbf'),
        'K Neighbours': KNeighborsClassifier(),
        '1 Layer NN (100)': MLPClassifier(),
        '2 Layer NN (100,50)': MLPClassifier(hidden_layer_sizes=(100,50)),
        '3 Layer NN (100,50,25)': MLPClassifier(hidden_layer_sizes=(100,50,25)),
        'AdaBoost': AdaBoostClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Random Forest': RandomForestClassifier()
    }

    # Train the algorithms on the data
    trained_networks = list()
    for clf_name, clf in classifier_algorithms.items():
        # Train the classifier and return the network and the scores in a dict
        training_output = hr.train_classifier(clf, ScaledTrainInput, TrainTarget)
        # Add the network name into the returned dict and append to trained_networks
        training_output.update({'Network Type': clf_name})
        trained_networks.append(training_output)
        #Evaluate the classifier performance and print the classification report
        Predictions = training_output['trained_clf'].predict(ScaledTestInput)
        print("Using the {} classifier:".format(clf_name))
        print("Accuracy: {:.2f}".format((TestTarget == Predictions).sum()/len(Predictions)) )
        print(classification_report(TestTarget, Predictions))
        print("")

# Part 2: Optimising the best algorithms further (random forest, 2 layer NN and gradient boost)
if input("Perform learning algorithm optimisation? (y/n): ") == 'y':
    # Optimise the random forest classifier
    clf = RandomForestClassifier()
    parameter_grid = {
        'n_estimators':[i for i in range(1,20,2)],
        'max_features':[2,4,6]
    }
    hr.optimise_classifier_f1(clf, parameter_grid, ScaledTrainInput, TrainTarget)
    
    # Optimise the 2-layer neural network
    clf = MLPClassifier()
    parameter_grid = {
        'hidden_layer_sizes':[(i,j) for i in [5,50,500] for j in [5,50,500]],
        'alpha':[0.0001, 1, 10, 1000]
    }
    hr.optimise_classifier_f1(clf, parameter_grid, ScaledTrainInput, TrainTarget)
    
    # Optimise the gradient boosting classifier
    clf = GradientBoostingClassifier()
    parameter_grid = {
        'n_estimators':[10, 20, 40, 80, 120, 160],
        'max_features':[i for i in range(1,10)]
    }
    hr.optimise_classifier_f1(clf, parameter_grid, ScaledTrainInput, TrainTarget)

# Part 3: Finalising the analysis of the random forest classifier
if input("Perform final analysis of random forest classifier? (y/n): ") == 'y':
    # Configure the random forest classifier using the optimal hyperparameters
    clf = RandomForestClassifier(
        n_estimators=13,
        max_features=2
        )
    # Train the classifier on the training data
    clf.fit(TrainInput,TrainTarget)
    # Compute the preidctions and probablities from the test data
    TestPrediction = clf.predict(TestInput)
    TestProbability = clf.predict_proba(TestInput)[:, 1]
    # Compute and plot the confusion matrix
    ConfMatrix = confusion_matrix(TestTarget,TestPrediction)
    hr.plot_confusion_matrix(ConfMatrix, ['Stayed','Left'])
    # Plot the ROC curve
    hr.plot_ROC_curve(TestTarget,TestProbability)