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
		'hidden_layer_sizes':[(25*i,25*j) for i in range(1,11) for j in range(1,11)]
	}
	hr.optimise_classifier_f1(clf, parameter_grid, ScaledTrainInput, TrainTarget)
	
	# Optimise the gradient boosting classifier
	clf = GradientBoostingClassifier()
	parameter_grid = {
		'n_estimators':[i for i in range(1,20,2)],
		'max_features':[2,4,6]
	}
	hr.optimise_classifier_f1(clf, parameter_grid, ScaledTrainInput, TrainTarget)