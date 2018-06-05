import numpy as np
import networkx as nx
import pandas as pd

from os import path
from graph_tool import load_graph
from networkx.readwrite import pajek


full_path = path.dirname(path.abspath(__file__ + "/../"))

PAJEK_FORMAT = path.join(full_path + "/data/DaysAll.net")
GRAPHML_FORMAT = path.join(full_path + "/data/DaysAll.graphml")
FIGURES_PATH = path.join(full_path + "/figures/")
RESULTS_PATH = path.join(full_path + "/results/")

DEGREE_FILE = "degree_centrality.tsv"
CLOSENESS_FILE = "closeness_centrality.tsv"
BETWEENNESS_FILE = "betweenness_centrality.tsv"


# Load graph into memory
def load(graph_path):
    return pajek.read_pajek(graph_path)


def save(graph, save_path):
    pajek.write_pajek(graph, save_path)


# Load graphml format to be used with graph tool
def load_graphml_format(graph_path):
    return load_graph(graph_path)


def calculate_local_clustering(graph, save_file_name):
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

    df.to_csv(RESULTS_PATH + save_file_name,
              sep='\t',
              float_format='%.4f',
              index=False)

    print("Result is saved to {}".format(RESULTS_PATH + save_file_name))


# Parse Pajek format graph file (.net) and get word day occurrences
def get_word_attributes(graph_file_path):
    word_day_dict = {}
    with open(graph_file_path, 'r') as graph:
        _, num_of_nodes = graph.readline().split()
        for idx, line in enumerate(graph):
            if idx == int(num_of_nodes):
                break

            vals = line.split()
            word_chars = list(vals[1])
            vals[1] = "".join(word_chars[1:len(word_chars)-1])
            if len(vals) == 5:
                word_day_dict[vals[1]] = []
            elif len(vals) == 6:
                days = eval(vals[5])

                if any(day < 0 for day in days):
                    word_day_dict[vals[1]] = list(range(1, abs(days[0]) + 2))
                else:
                    word_day_dict[vals[1]] = days

    return word_day_dict
