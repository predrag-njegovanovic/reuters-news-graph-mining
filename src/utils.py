import numpy as np
import networkx as nx
import pandas as pd

from os import path
from graph_tool import load_graph
from graph_tool import clustering
from networkx.readwrite import pajek


full_path = path.dirname(path.abspath(__file__ + "/../"))

PAJEK_FORMAT = path.join(full_path + "/data/DaysAll.net")
GRAPHML_FORMAT = path.join(full_path + "/data/DaysAll.graphml")
FIGURES_PATH = path.join(full_path + "/figures/")
RESULTS_PATH = path.join(full_path + "/results/")


# Load graph into memory
def load(graph_path):
    return pajek.read_pajek(graph_path)


def save(graph, save_path):
    pajek.write_pajek(graph, save_path)


# Load graphml format to be used with graph tool
def load_graphml_format(graph_path):
    return load_graph(graph_path)


def calculate_local_clustering(graph, file_name):
    print("Calculating words local clustering coefficient...")
    local_clustering = nx.clustering(graph)
    cluster_values = list(map(lambda coeff: round(coeff, 4), local_clustering.values()))

    word_coeff = np.concatenate((np.array(list(local_clustering.keys()),
                                          ndmin=2).T,
                                 np.array(cluster_values,
                                          ndmin=2).T),
                                axis=1)

    df = pd.DataFrame(data=word_coeff,
                      columns=["Word", "Local clustering"])

    df.to_csv(RESULTS_PATH + file_name,
              sep='\t',
              float_format='%.4f',
              index=False)

    print("Result is saved to {}".format(RESULTS_PATH + file_name))
