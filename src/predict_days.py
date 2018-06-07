import pandas as pd
import numpy as np

from src.utils import DATASET_PATH


def load_dataset(dataset_path):
    data = pd.read_csv(dataset_path, sep='\t').values
    col_num = data.shape[1]
    return data[:, 0:col_num-1], data[:, col_num-1]


def analyze_dataset(dataset_path):
    return None
