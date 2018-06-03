import operator
import numpy as np
import networkx as nx
import pandas as pd

from src.utils import load
from src.utils import PAJEK_FORMAT
from src.utils import RESULTS_PATH


def degree_centrality(graph):
    return ("degree_centrality.tsv", nx.degree_centrality(graph))


def closeness_centrality(graph):
    return ("closeness_centrality.tsv", nx.closeness_centrality(graph))


def betweenness_centrality(graph):
    return ("betweenness_centrality.tsv", nx.betweenness_centrality(graph, weight="weight"))


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


if __name__ == "__main__":
    G = load(PAJEK_FORMAT).to_undirected()
    centrality_tuple = betweenness_centrality(G)
    most_influential_words(centrality_tuple[0], centrality_tuple[1])
