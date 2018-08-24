#!/usr/bin/env bash
mkdir -p ./data
mkdir -p ./data/original
curl https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene.param --output ./data/original/arcene.param
curl https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_test.data --output ./data/original/test.data
curl https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.data --output ./data/original/train.data
curl https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.labels --output ./data/original/train.labels
curl https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_valid.data --output ./data/original/valid.data
curl https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/arcene_valid.labels --output ./data/original/valid.labels
