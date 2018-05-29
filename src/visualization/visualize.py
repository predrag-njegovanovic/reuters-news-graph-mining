import networkx as nx
import matplotlib.pyplot as plt
import graph_tool.draw as gd
import collections

from src.utils import load
from src.utils import load_graphml_format
from src.utils import PAJEK_FORMAT
from src.utils import GRAPHML_FORMAT
from src.utils import FIGURES_PATH

SAVED_GRAPH_NAME = "ReutersNews"
DEGREE_DISTRIBUTION = "DegreeDistrib.png"
DEGREE_THRESHOLD = 200


def visualize_graph(reuters_graph):
    pos = gd.fruchterman_reingold_layout(reuters_graph)
    print("Graph visualization running...")
    gd.graph_draw(reuters_graph, pos=pos)
    print("Visualization finished...")
    reuters_graph.save(FIGURES_PATH + SAVED_GRAPH_NAME, fmt="graphml")


def draw_degree_distribution(reuters_graph):
    degrees = filter(lambda nd_view: nd_view[1] <= DEGREE_THRESHOLD,
                     reuters_graph.degree)
    degrees = sorted([d for node, d in degrees])
    degree_count = collections.Counter(degrees)
    deg, count = zip(*degree_count.items())
    plt.bar(deg, count, color='r')
    plt.title("Degree distribution")
    plt.xlabel("Degree")
    plt.ylabel("Words count")
    plt.savefig(FIGURES_PATH + DEGREE_DISTRIBUTION)
    plt.show()
