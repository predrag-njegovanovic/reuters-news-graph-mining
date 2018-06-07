import numpy as np
import networkx as nx
import pandas as pd

from os import path
from graph_tool import load_graph
from networkx.readwrite import pajek


full_path = path.dirname(path.abspath(__file__ + "/../"))

DATASET_PATH = path.join(full_path + "/data/dataset.tsv")
PAJEK_FORMAT = path.join(full_path + "/data/DaysAll.net")
GRAPHML_FORMAT = path.join(full_path + "/data/DaysAll.graphml")
FIGURES_PATH = path.join(full_path + "/figures/")
RESULTS_PATH = path.join(full_path + "/results/")

DEGREE_FILE = "degree_centrality.tsv"
CLOSENESS_FILE = "closeness_centrality.tsv"
BETWEENNESS_FILE = "betweenness_centrality.tsv"

CLUSTERTING_COLUMN = "Local clustering"
DEGREE_COLUMN = "Node degree"
CLOSENESS_COLUMN = "Closeness coefficient"


# Load graph into memory
def load(graph_path):
    return pajek.read_pajek(graph_path)


def save(graph, save_path):
    pajek.write_pajek(graph, save_path)


# Load graphml format to be used with graph tool
def load_graphml_format(graph_path):
    return load_graph(graph_path)


# Parse Pajek format graph file (.net) and get word-day occurrence dictionary
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


def create_dataset(cluster_df, degree_df, closeness_df, graph_path=PAJEK_FORMAT):
    data_df = pd.concat([cluster_df['Local clustering'],
                         degree_df['Node degree'],
                         closeness_df['Closeness coefficient']], axis=1)

    target_values = []
    word_day_dict = get_word_attributes(PAJEK_FORMAT)
    for value in word_day_dict.values():
        target_values.append(len(value))

    data_df['Number of days'] = target_values
    data_df.to_csv(DATASET_PATH,
                   sep='\t',
                   index=False)
    print("Dataset saved...")


def calculate_local_clustering(graph, save_file_name):
    print("Calculating words local clustering coefficient...")
    local_clustering = nx.clustering(graph)
    save_features(local_clustering, CLUSTERTING_COLUMN, save_file_name)


def calculate_node_degree(graph, save_file_name):
    node_degree = graph.degree(weight="weight")
    save_features(dict(node_degree), DEGREE_COLUMN, save_file_name)


# Save feature to file using node-feature dictionary
def save_features(feature_dict, feature_name, save_file_name):
    feature_values = list(map(lambda coeff: round(coeff, 4), feature_dict.values()))

    word_coeff = np.concatenate((np.array(list(feature_dict.keys()),
                                          ndmin=2).T,
                                 np.array(feature_values,
                                          ndmin=2).T),
                                axis=1)

    df = pd.DataFrame(data=word_coeff,
                      columns=["Word", feature_name])

    df.to_csv(RESULTS_PATH + save_file_name,
              sep='\t',
              float_format='%.4f',
              index=False)

    print("Result is saved to {}".format(RESULTS_PATH + save_file_name))


def read_features(clustering_file_name, node_file_name, closeness_file_name):
    graph = nx.Graph(load(PAJEK_FORMAT)).to_undirected()
    try:
        print("Reading local clustering features...")
        cluster_df = pd.read_csv(RESULTS_PATH + clustering_file_name,
                                 sep='\t')
    except FileNotFoundError:
        print("Local clustering features file doesn't exist.")
        calculate_local_clustering(graph, clustering_file_name)
        cluster_df = pd.read_csv(RESULTS_PATH + clustering_file_name,
                                 sep='\t')

    try:
        print("Reading node degree features...")
        degree_df = pd.read_csv(RESULTS_PATH + node_file_name,
                                sep='\t')
    except FileNotFoundError:
        print("Node degree features file doesn't exist.")
        calculate_node_degree(graph, node_file_name)
        degree_df = pd.read_csv(RESULTS_PATH + node_file_name,
                                sep='\t')

    try:
        print("Reading closeness centrality features...")
        closeness_df = pd.read_csv(RESULTS_PATH + closeness_file_name,
                                   sep='\t')
    except FileNotFoundError:
        print("Closeness centrality feature file doesn't exist.")
        save_features(nx.closeness_centrality(graph),
                      CLOSENESS_COLUMN,
                      closeness_file_name)
        closeness_df = pd.read_csv(RESULTS_PATH + closeness_file_name,
                                   sep='\t')

    return cluster_df, degree_df, closeness_df
