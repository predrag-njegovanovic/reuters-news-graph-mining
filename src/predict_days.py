import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats.stats import spearmanr
from src.utils import DATASET_PATH


def load_dataset(dataset_path):
    data = pd.read_csv(dataset_path, sep='\t').values
    col_num = data.shape[1]
    return data[:, 0:col_num-1], data[:, col_num-1]


def plot_feature_distributions(features):
    sns.distplot(features[:, 0])
    plt.title("Local clustering distribution")
    plt.show()

    sns.distplot(features[:, 1])
    plt.title("Node degree distribution")
    plt.show()

    sns.distplot(features[:, 2])
    plt.title("Closeness centrality distribution")
    plt.show()


def features_correlation(features):
    corr = spearmanr(features)[0]
    sns.heatmap(corr,
                annot=True,
                linewidth=.5,
                xticklabels=["Local clustering", "Node degree", "Closeness"],
                yticklabels=["Local clustering", "Node degree", "Closeness"])
    plt.show()
