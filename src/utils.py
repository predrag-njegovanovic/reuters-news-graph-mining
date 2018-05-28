from os import path
from graph_tool import load_graph

FULL_PATH = path.dirname(path.abspath(__file__ + "/../"))
GRAPH_PATH = path.join(FULL_PATH + "/data/Days.graphml")


# Load graph into memory
def load(graph_path):
    return load_graph(graph_path)


def save(graph, save_path):
    graph.save(save_path)
