import operator
import numpy as np
import networkx as nx
import pandas as pd
import sys

from src.utils import load
from src.utils import get_word_attributes
from src.utils import PAJEK_FORMAT
from src.utils import RESULTS_PATH


def degree_centrality(graph):
    return nx.degree_centrality(graph)


def closeness_centrality(graph):
    return nx.closeness_centrality(graph)


def betweenness_centrality(graph):
    return nx.betweenness_centrality(graph, weight="weight")


def most_influential_words(centrality_name, centrality_measure):
    word_dict = sorted(centrality_measure.items(),
                       key=operator.itemgetter(1),
                       reverse=True)
    word_dict = list(map(lambda tup: (tup[0], round(tup[1], 4)), word_dict))
    influential_words = word_dict[:10]
    noninfluent_words = word_dict[-10:]
    df_data = np.concatenate((np.array(influential_words),
                              np.array(noninfluent_words)), axis=1)

    df = pd.DataFrame(data=df_data,
                      columns=["Influential words",
                               "Influence",
                               "Noninfluential words",
                               "Influence"])
    df.to_csv(RESULTS_PATH + centrality_name,
              sep='\t',
              float_format='%.4f',
              index=False)

    print_function = lambda wm_pair: print("Word: {} ----> Influence: {:.4f}".format(wm_pair[0], wm_pair[1]))
    print("Top 10 most influential words:\n")
    for wm_pair in influential_words:
        print_function(wm_pair)
    print("<----------------------------------->\n")
    print("Top 10 lowest influential words:\n")
    for wm_pair in noninfluent_words:
        print_function(wm_pair)


def word_day_occurrences(centrality_file_path=None,
                         centrality_measure='degree',
                         graph_path=PAJEK_FORMAT):

    if(centrality_file_path is None):
        print("Trying to load graph from file path: {}".format(graph_path))
        try:
            graph = load(graph_path)
        except FileNotFoundError:
            print('File not found on specified path.')
            sys.exit(1)

        if centrality_measure == 'degree':
            words = degree_centrality(graph).keys()
        elif centrality_measure == 'closeness':
            words = closeness_centrality(graph).keys()
        elif centrality_measure == 'betweenness':
            words = betweenness_centrality(graph).keys()
        else:
            print('Specified centrality measure not implemented.')
            sys.exit(1)
    else:
        centrality_data = pd.read_csv(centrality_file_path, sep='\t')
        words = list(centrality_data.iloc[:, 0])
        word_day_dict = get_word_attributes(graph_path)
        for word in words:
            print("Word: {},  Days of occurrence: {}\n".format(word, word_day_dict[word]))
