import operator
import networkx as nx

from src.utils import load
from src.utils import PAJEK_FORMAT


def degree_centrality(graph):
    return nx.degree_centrality(graph)


def closeness_centrality(graph):
    return nx.closeness_centrality(graph)


def betweenness_centrality(graph):
    return nx.betweenness_centrality(graph, weight="weight")


def most_influential_words(centrality_measure):
    word_dict = sorted(centrality_measure.items(),
                       key=operator.itemgetter(1),
                       reverse=True)
    influential_words = word_dict[:10]
    noninfluent_words = word_dict[-10:]

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
