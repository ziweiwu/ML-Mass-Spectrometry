import urllib
import os


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)


# Example
createFolder('./data/original')

# Creates a folder in the current directory called data
urllib.("https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene.param",
                  "./data/original/arcene.param")
