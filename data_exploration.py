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


# convert data files to csv format
convert_to_csv("./data/original/train.data", "./data/train.csv")
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

# load the dataset
X = pd.read_csv("data/train.csv")
y = pd.read_csv("data/train_labels.csv")

# basic overview of data dimension
print(X.shape)
print(y.shape)

y_data = y.values.flatten()
X_data = X

# use TSNE to visualize the high dimension data in 2D
t0 = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=5000, random_state=100)
tsne_results = tsne.fit_transform(X_data)
t1 = time.time()
print("TSNE took at %.2f seconds" % (t1 - t0))

# visualize TSNE and save the plot
x_axis = tsne_results[:, 0]
y_axis = tsne_results[:, 1]
plt.scatter(x_axis, y_axis, c=y_data, cmap=plt.cm.get_cmap("jet", 100))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.title("TSNE Visualization")
plt.savefig("./images/tsne_graph.png", dpi=600)
