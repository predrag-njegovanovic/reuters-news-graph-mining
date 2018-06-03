import matplotlib
import matplotlib.pyplot as plt
import graph_tool.draw as gd
import graph_tool.centrality as gc
import graph_tool.inference as gi
import collections

from src.utils import load_graphml_format
from src.utils import GRAPHML_FORMAT
from src.utils import FIGURES_PATH

SAVED_GRAPH_NAME = "reuters_news"
GRAPH_VIZ = "reuters_news.png"
DEGREE_DISTRIBUTION = "degree_distribution.png"
BETWEENNESS_VIZ = "betweenness_influent_nodes.png"
CLOSENESS_VIZ = "closeness_influent_nodes.png"
COMMUNITY_VIZ = "community_visualization.png"
DEGREE_THRESHOLD = 200


def visualize_graph(reuters_graph):
    pos = gd.fruchterman_reingold_layout(reuters_graph)
    print("Graph visualization running...")
    gd.graph_draw(reuters_graph,
                  pos=pos,
                  output_size=(1920, 1080),
                  output=FIGURES_PATH + GRAPH_VIZ)
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


def betweenness_influent_nodes(graph):
    vertex_prop, edge_prop = gc.betweenness(graph,
                                            weight=graph.edge_properties["weight"])
    gd.graph_draw(graph, vertex_fill_color=vertex_prop,
                  vertex_size=gd.prop_to_size(vertex_prop, mi=5, ma=15),
                  edge_pen_width=gd.prop_to_size(edge_prop, mi=0.5, ma=5),
                  vcmap=matplotlib.cm.gist_heat, vorder=vertex_prop,
                  output_size=(1920, 1080),
                  output=FIGURES_PATH + BETWEENNESS_VIZ)

    print("Visualization finished and saved...")


def closeness_influent_nodes(graph):
    vertex_prop = gc.closeness(graph)
    gd.graph_draw(graph, vertex_fill_color=vertex_prop,
                  vertex_size=gd.prop_to_size(vertex_prop, mi=5, ma=15),
                  vcmap=matplotlib.cm.gist_heat, vorder=vertex_prop,
                  output_size=(1920, 1080),
                  output=FIGURES_PATH + CLOSENESS_VIZ)

    print("Visualization finished and saved...")


def visualize_communities(graph):
    state_block = gi.minimize_nested_blockmodel_dl(graph,
                                                   B_min=5,
                                                   B_max=10,
                                                   verbose=True)
    gd.draw_hierarchy(state_block,
                      vertex_text=graph.vertex_properties["_graphml_vertex_id"],
                      vertex_size=1,
                      vertex_font_size=6,
                      output_size=(1920, 1080),
                      layout="sfdp",
                      output=FIGURES_PATH + COMMUNITY_VIZ)


if __name__ == "__main__":
    G = load_graphml_format(GRAPHML_FORMAT)
    G.set_directed(False)
    visualize_communities(G)
