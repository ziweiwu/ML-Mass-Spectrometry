import os
import time
import json
import pandas as pd
import numpy as np
from sklearn import svm, ensemble, linear_model
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.metrics import recall_score

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

# define the models
logit = linear_model.LogisticRegression(random_state=100, dual=False)
linear_svm = svm.LinearSVC(random_state=100, dual=False)
none_linear_svm = svm.SVC(random_state=100)
rf = ensemble.RandomForestClassifier(random_state=100)
nn = MLPClassifier(random_state=100)


#################################################################################
# test the models before parameter tuning
#################################################################################
def model_recall_test(model, model_name):
    t0 = time.time()
    model = model.fit(X_train, y_train)
    t1 = time.time()
    padding = 16
    recall = recall_score(y_true=y_test, y_pred=model.predict(X_test), average="macro")
    print("{}      {:0.4}      {:0.4}".format(model_name.ljust(padding), (t1 - t0), recall))


print("Model         Training time     Recall score")
model_recall_test(logit, "Logistic regression ")
model_recall_test(linear_svm, "Linear svm")
model_recall_test(none_linear_svm, "None-linear svm")
model_recall_test(rf, "Random forest")
model_recall_test(nn, "Neuron network ")

#################################################################################
#  Perform  parameter tuning
#################################################################################
print("Parameter tuning starts...")


def model_tune_params(model, params):
    if __name__ == '__main__':
        new_model = GridSearchCV(estimator=model,
                                 param_grid=params, cv=5, n_jobs=-1,
                                 scoring="recall_macro")
        new_model.fit(X_train, y_train)
        print(new_model.best_params_, '\n')
        return new_model


logit_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'penalty': ('l2', 'l1')
}

linear_svm_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'penalty': ('l2', 'l1')
}

none_linear_svm_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    'gamma': [0.001, 0.0001, 0.00001],
    'kernel': ('poly', 'rbf', 'sigmoid')
}
rf_params = {
    'n_estimators': [10, 20, 30, 40, 50],
    'max_leaf_nodes': [50, 100, 150, 200],
    'min_samples_split': [2, 3, 10],
    'min_samples_leaf': [1, 3, 10],
    'bootstrap': [True],
    'criterion': ['gini', 'entropy']
}

nn_params = {
    'hidden_layer_sizes': [50, 100, 200, 500],
    'alpha': [0.0001, 0.0005, 0.001, 0.005],
    'activation': ('relu', 'tanh', 'identity'),
}

logit = model_tune_params(logit, logit_params)
linear_svm = model_tune_params(linear_svm, linear_svm_params)
none_linear_svm = model_tune_params(none_linear_svm, none_linear_svm_params)
rf = model_tune_params(rf, rf_params)
nn = model_tune_params(nn, nn_params)

#################################################################################
# test the models after parameter tuning
#################################################################################
# save the models
if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(logit, "models/logit.pkl", compress=3)
joblib.dump(linear_svm, "models/linear_svm.pkl", compress=3)
joblib.dump(none_linear_svm, "models/none_linear_svm.pkl", compress=3)
joblib.dump(rf, "models/rf.pkl", compress=3)
joblib.dump(nn, "models/nn.pkl", compress=3)
print("Models saved.")
