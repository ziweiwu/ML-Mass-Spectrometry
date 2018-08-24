import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split


def convert_to_csv(input, output):
    with open(input) as fin, open(output, 'w') as fout:
        o = csv.writer(fout)
        for line in fin:
            o.writerow(line.split())


# convert data files to csv format convert_to_csv("./data/original/train.data", "./data/train.csv")
convert_to_csv("./data/original/train.labels", "./data/train_labels.csv")
convert_to_csv("./data/original/test.data", "./data/test.csv")
convert_to_csv("./data/original/test.data", "./data/test.csv")
convert_to_csv("./data/original/valid.data", "./data/valid.csv")
convert_to_csv("./data/original/valid.labels", "./data/valid_labels.csv")


def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


# make some dictionaries
make_directory("images")
make_directory("models")
make_directory("./data/train")
make_directory("./data/test")

# load the dataset
X = pd.read_csv("data/train.csv")
y = pd.read_csv("data/train_labels.csv")

# basic overview of data dimension
print(X.shape)
print(y.shape)

X_data = X.values
y_data = y.values.flatten()

print("Creating tsne visualizations...")


def tsne_visualization():
    PERLEXITY = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    fig, axes = plt.subplots(5, 2, figsize=(10, 20), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.2)
    for i in range(10):
        ax = axes.flatten()[i]
        tsne = TSNE(n_components=2, verbose=1, perplexity=PERLEXITY[i], n_iter=5000, random_state=100)
        tsne_results = tsne.fit_transform(X_data)
        x_axis = tsne_results[:, 0]
        y_axis = tsne_results[:, 1]
        ax.scatter(x_axis, y_axis)
        ax.set_title("With perplexity {}".format(PERLEXITY[i]))


# use TSNE to visualize the high dimension data in 2D
tsne_visualization()

# Shuttle the data and split it into training and test set
X_train, X_test, y_train, y_test \
    = train_test_split(X, y, test_size=0.20, random_state=100, stratify=y, shuffle=True)

# save the train and test csv files
X_train.to_csv("./data/train/X_train.csv")
y_train.to_csv("./data/train/y_train.csv")
X_test.to_csv("./data/test/X_test.csv")
y_test.to_csv("./data/test/y_test.csv")
