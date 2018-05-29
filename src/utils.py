from os import path
from graph_tool import load_graph
from networkx.readwrite import pajek

full_path = path.dirname(path.abspath(__file__ + "/../"))

PAJEK_FORMAT = path.join(full_path + "/data/DaysAll.net")
GRAPHML_FORMAT = path.join(full_path + "/data/DaysAll.graphml")
FIGURES_PATH = path.join(full_path + "/figures/")


# Load graph into memory
def load(graph_path):
    return pajek.read_pajek(graph_path)


def save(graph, save_path):
    pajek.write_pajek(graph, save_path)


# Load graphml format to be used with graph tool
def load_graphml_format(graph_path):
    return load_graph(graph_path)
