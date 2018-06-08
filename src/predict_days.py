import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats.stats import spearmanr
from src.utils import DATASET_PATH
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def load_dataset(dataset_path):
    data = pd.read_csv(dataset_path, sep='\t').values
    col_num = data.shape[1]
    return data[:, 0:col_num-1], data[:, col_num-1]


def normalize_features(features):
    features_mean = np.mean(features, axis=0)
    features_std = np.std(features, axis=0)
    for i in range(features.shape[1]):
        features[:, i] = (features[:, i] - features_mean[i])/features_std[i]

    return features, features_mean, features_std


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


def fit_model(features, labels):
    model = LassoCV(fit_intercept=False,
                    n_jobs=-1,
                    n_alphas=200,
                    max_iter=5000)
    model.fit(features, labels)
    return model


if __name__ == "__main__":
    features, labels = load_dataset(DATASET_PATH)
    plot_feature_distributions(features)
    features_correlation(features)
    poly = PolynomialFeatures(degree=2)
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=0.2)
    X_train, mean, std = normalize_features(X_train)
    for i in range(X_test.shape[1]):
        X_test[:, i] = (X_test[:, i] - mean[i])/std[i]

    X_train = poly.fit_transform(X_train)
    model = fit_model(X_train, y_train)
    X_test = poly.fit_transform(X_test)
    y_pred = np.rint(model.predict(X_test))
    print("Model R2 score: {}".format(model.score(X_test, y_test)))
