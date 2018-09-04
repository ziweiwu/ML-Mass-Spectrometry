import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

# load the training data
print("Loading data sets...")
X_train = pd.read_csv("data/train/X_train.csv")
y_train = pd.read_csv("data/train/y_train.csv")
X_test = pd.read_csv("data/test/X_test.csv")
y_test = pd.read_csv("data/test/y_test.csv")

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


def load_model(path):
    return pickle.load(open(path, "rb"))


# load models
logit = load_model("models/logit.pkl")
linear_svm = load_model("models/linear_svm.pkl")
none_linear_svm = load_model("models/none_linear_svm.pkl")
rf = load_model("models/rf.pkl")
nn = load_model("models/nn.pkl")
print("Models loaded")


def get_cv_scores(scoring="recall_macro", cv=5):
    cv_scores = []
    cv_scores.append(cross_val_score(logit, X_train, y_train, scoring=scoring, cv=cv).mean())
    cv_scores.append(cross_val_score(linear_svm, X_train, y_train, scoring=scoring, cv=cv).mean())
    cv_scores.append(cross_val_score(none_linear_svm, X_train, y_train, scoring=scoring, cv=cv).mean())
    cv_scores.append(cross_val_score(rf, X_train, y_train, scoring=scoring, cv=cv).mean())
    cv_scores.append(cross_val_score(nn, X_train, y_train, scoring=scoring, cv=cv).mean())
    return cv_scores


#################################################################################
# Compare model performance using CV
#################################################################################
plt.bar(x=['logit', 'l-svm', 'nl-svm', 'rf', 'nn'], height=get_cv_scores())
plt.xlabel('Models')
plt.ylabel('Recall')
plt.title("Model Comparison")
plt.savefig('images/models_comparison.png', dpi=600)
