import matplotlib.pyplot as plt
from sklearn import svm, ensemble, linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import recall_score
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np

########################################################################################
#                    Load dataset
########################################################################################
# load the training data
print("Loading data sets...")
X_train = pd.read_csv("./data/train/X_train.csv")
y_train = pd.read_csv("./data/train/y_train.csv")
X_test = pd.read_csv("./data/test/X_test.csv")
y_test = pd.read_csv("./data/test/y_test.csv")

# transform panda df into arrays
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values.flatten()
y_test = y_test.values.flatten()

index_to_delete = []
for i in range(len(y_train)):
    if i % 2 == 0:
        index_to_delete.append(i)

y_train = np.delete(y_train, index_to_delete)
y_test = np.delete(y_test, index_to_delete)

print(X_train)
print(y_train)

print(X_train.shape)
print(y_train.shape)

print("Dataset loaded.")


########################################################################################
#                    Feature selection
########################################################################################
# param set for grid search for each model
def model_tune_params(model, params):
    new_model = GridSearchCV(estimator=model,
                             param_grid=params, cv=5,
                             scoring="recall_macro", n_jobs=-1)
    return new_model


sgd_params = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'penalty': ('l2', 'l1')
}

logit_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'penalty': ('l2', 'l1')
}

linear_svm_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'penalty': ('l2', 'l1')
}

rf_params = {
    'n_estimators': [10, 20, 30, 40, 50],
    'max_leaf_nodes': [50, 100, 150, 200],
    'min_samples_split': [2, 3, 10],
    'min_samples_leaf': [1, 3, 10],
    'bootstrap': [True],
    'criterion': ['gini', 'entropy']
}

n_features_sgd = []
recall_sgd = []

# feature selection
alpha_params = [0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.1,
                1, 10, 100, 200, 300, 400, 500, 550, 600, 620, 640, 660, 680, 700]


# perform feature selection using sparse svm
def sgd_feature_selection(alpha_params):
    for alpha in alpha_params:
        est = linear_model.SGDClassifier(random_state=100, penalty="l1", alpha=alpha, tol=1e-3)
        transformer = SelectFromModel(estimator=est)
        train_features = transformer.fit_transform(X_train, y_train)
        test_features = transformer.transform(X_test)
        print("\nWith alpha={}".format(alpha))
        print("SGD reduced number of features to {}.".format(test_features.shape[1]))

        model = linear_model.SGDClassifier(random_state=100, tol=1e-3)
        if test_features.shape[1] <= 1000:
            model = model_tune_params(model, sgd_params)
        model.fit(train_features, y_train)
        score = recall_score(y_pred=model.predict(test_features), y_true=y_test, average="macro")
        print("SGD recall after FEATURE SELECTION: {:5f}".format(score))
        n_features_sgd.append(test_features.shape[1])
        recall_sgd.append(score)


sgd_feature_selection(alpha_params)

n_features_svm = []
recall_svm = []
C_params = [0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.1, 1,
            10, 100, 1000, 10000, 100000, 1000000]
C_params.reverse()


# perform feature selection using sparse svm
def svm_feature_selection(C_params):
    for C in C_params:
        est = svm.LinearSVC(random_state=100, penalty="l1", C=C, dual=False, tol=1e-4)
        transformer = SelectFromModel(estimator=est)
        train_features = transformer.fit_transform(X_train, y_train)
        test_features = transformer.transform(X_test)
        print("\nWith C={}".format(C))
        print("Sparse SVM reduced number of features to {}.".format(test_features.shape[1]))

        model = svm.LinearSVC(random_state=100, dual=False)
        if test_features.shape[1] <= 1000:
            model = model_tune_params(model, linear_svm_params)
        model.fit(train_features, y_train)
        score = recall_score(y_pred=model.predict(test_features), y_true=y_test, average="macro")
        print("Linear SVC recall after FEATURE SELECTION: {:5f}".format(score))
        n_features_svm.append(test_features.shape[1])
        recall_svm.append(score)


svm_feature_selection(C_params)

# perform feature selection using rf, use mean as threshold
thresholds = [0, 1e-06, 2e-06, 5e-06, 1e-05, 2e-05, 5e-05, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004,
              0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011]

n_features_rf = []
recall_rf = []


def rf_feature_selection(thresholds):
    for threshold in thresholds:
        est = ensemble.RandomForestClassifier(random_state=100, n_estimators=50, n_jobs=-1)
        transformer = SelectFromModel(estimator=est, threshold=threshold)
        train_features = transformer.fit_transform(X_train, y_train)
        test_features = transformer.transform(X_test)
        print("\nWith threshold {}".format(threshold))
        print("RF reduced number of features to {}.".format(test_features.shape[1]))

        model = ensemble.RandomForestClassifier(random_state=100)
        if test_features.shape[1] <= 1000:
            model = model_tune_params(model, rf_params)
        model.fit(train_features, y_train)
        score = recall_score(y_pred=model.predict(test_features), y_true=y_test, average="macro")
        print("RF recall after FEATURE SELECTION: {:5f}".format(score))
        n_features_rf.append(test_features.shape[1])
        recall_rf.append(score)


rf_feature_selection(thresholds)

# perform feature selection using logistic regression
C_params = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01,
            10, 100, 1000, 10000, 100000, 1000000]
C_params.reverse()

n_features_logit = []
recall_logit = []


def logit_feature_selection(C_params):
    for C in C_params:
        est = linear_model.LogisticRegression(random_state=100, penalty="l1", C=C, tol=1e-4)
        transformer = SelectFromModel(estimator=est)
        train_features = transformer.fit_transform(X_train, y_train)
        test_features = transformer.transform(X_test)
        print("\nWith C={}".format(C))
        print("Logistic regression reduced number of features to {}.".format(test_features.shape[1]))

        model = linear_model.LogisticRegression(random_state=100)
        if test_features.shape[1] <= 1000:
            model = model_tune_params(model, logit_params)
        model.fit(train_features, y_train)
        score = recall_score(y_pred=model.predict(test_features), y_true=y_test, average="macro")
        print("Logistic regression recall after FEATURE SELECTION: {:5f}".format(score))
        n_features_logit.append(test_features.shape[1])
        recall_logit.append(score)


logit_feature_selection(C_params)

########################################################################################
#                    Feature Selection Performance
########################################################################################
print(n_features_svm)
print(recall_svm)
print(n_features_rf)
print(recall_rf)
print(n_features_logit)
print(recall_logit)

figure(num=None, figsize=(8, 8), dpi=800, facecolor='w', edgecolor='k')
plt.xlabel('Number of Features')
plt.ylabel('Recall')
plt.title("Number of Features vs Recall")
plt.plot(n_features_sgd, recall_sgd, 'o-', color='orange')
plt.plot(n_features_svm, recall_svm, 'o-', color='blue')
plt.plot(n_features_rf, recall_rf, 'o-', color='green')
plt.plot(n_features_logit, recall_logit, 'o-', color='red')
plt.legend(['SGD', 'SVM', 'Random Forest', 'Logistic Regression'], loc=5)
plt.axis([0, 1000, 0.5, 1])
plt.savefig('images/feature_selection_performance.png', dpi=600)
