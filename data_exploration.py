import time
import csv
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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


# function to make
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


# make directory to store images
mkdir_p("./images")

# make directory to store model
mkdir_p("./models")

# load the dataset
X = pd.read_csv("data/train.csv")
y = pd.read_csv("data/train_labels.csv")

# basic overview of data dimension
print(X.shape)
print(y.shape)

X_data = X.values
y_data = y.values.flatten()


def tsne_visualization():
    PERLEXITY = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    fig, axes = plt.subplots(5, 2, figsize=(20, 40), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.1, wspace=.1)
    for i in range(10):
        ax = axes.flatten()[i]
        tsne = TSNE(n_components=2, verbose=1, perplexity=PERLEXITY[i], n_iter=5000, random_state=100)
        tsne_results = tsne.fit_transform(X_data)
        x_axis = tsne_results[:, 0]
        y_axis = tsne_results[:, 1]
        ax.scatter(x_axis, y_axis, c=y_data)
        ax.set_title("With perplexity {}".format(PERLEXITY[i]))
    fig.savefig("./images/tsne_graph.png", dpi=300)


# use TSNE to visualize the high dimension data in 2D
tsne_visualization()


# Shuttle the data and split it into training and test set
X_train, X_test, y_train, y_test \
    = train_test_split(X, y, test_size=0.20, random_state=100, stratify=y, shuffle=True)

# save the train and test csv files
mkdir_p("./data/train")
mkdir_p("./data/test")
X_train.to_csv("./data/train/X_train.csv")
y_train.to_csv("./data/train/y_train.csv")
X_test.to_csv("./data/test/X_test.csv")
y_test.to_csv("./data/test/y_test.csv")
