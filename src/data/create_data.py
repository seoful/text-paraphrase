import pandas as pd
import glob
import os
from os import listdir
from os.path import isfile, join

path = 'data/raw/shards' # use your path
abs_path = os.path.abspath(path)

num = len(listdir(abs_path))

li = []

for i in range(num):
    df = pd.read_csv(abs_path + f'\chunk{i}.csv', index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

frame.to_csv('data/raw/dataset.csv')

path = 'data/interim/shards' # use your path
abs_path = os.path.abspath(path)

num = len(listdir(abs_path))

li = []

for i in range(num):
    df = pd.read_csv(abs_path + f'\chunk{i}.csv', index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

frame.to_csv('data/interim/data.csv')