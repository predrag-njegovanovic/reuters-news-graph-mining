import sys
import argparse
import src.visualization.visualize as viz
import src.graph_analysis as ga

from src.utils import load
from src.utils import load_graphml_format
from src.predict_days import load_dataset
from src.predict_days import predict_days
from src.utils import GRAPHML_FORMAT
from src.utils import PAJEK_FORMAT
from src.utils import DATASET_PATH


if __name__ == "__main__":
    reuters_graphml = load_graphml_format(GRAPHML_FORMAT)
    reuters_pajek = load(PAJEK_FORMAT)
    parser = argparse.ArgumentParser(description="Initializing graph analysis")
    parser.add_argument('-vg',
                        '--visualize-graph',
                        help="Visualize graph and save it to /figures folder")
    parser.add_argument('-ddd',
                        '--draw-degree-distribution',
                        help="Visualize degree distribution")
    parser.add_argument('-bc',
                        '--betweenness-centrality',
                        help="Visualize nodes with highest centrality value")
    parser.add_argument('-cc',
                        '--closeness-centrality',
                        help="Visualize nodes with highest centrality value")
    parser.add_argument('-vc',
                        '--visualize-communities',
                        help="Visualize graph communities")
    parser.add_argument('-inf-w',
                        '--influential-words',
                        nargs=2,
                        type=str,
                        help="Calculate word influence by utilizing centrality measure." \
                             " Params: centrality name," \
                                      "[\'degree\', \'betweenness\', \' closeness\']")
    parser.add_argument('-word-occ',
                        '--word-day-occurrences',
                        nargs='*',
                        type=str,
                        help="Show word day occurence." \
                                " Params: centrality_file_name [Optional], " \
                                          "centrality_measure: [\'degree\', \'betweenness\', \'closeness\']")
    parser.add_argument('-pd',
                        '--predict-days',
                        help="Run linear regression and predict number of occurrences")

    args = parser.parse_args()
    if args.visualize_graph:
        viz.visualize_graph(reuters_graphml)

    if args.draw_degree_distribution:
        print("===> Drawing degree distribution...")
        viz.draw_degree_distribution(reuters_pajek)

    if args.betweenness_centrality:
        print("===> Betweenness centrality visualization...")
        viz.betweenness_influent_nodes(reuters_graphml)

    if args.closeness_centrality:
        print("===> Closeness centrality visualization...")
        viz.closeness_influent_nodes(reuters_graphml)

    if args.visualize_communities:
        print("===> Visualizing communities...")
        viz.visualize_communities(reuters_graphml)

    if args.influential_words:
        if args.influential_words[1] == 'degree':
            centrality_measure = ga.degree_centrality(reuters_pajek)
        elif args.influential_words[1] == 'betweenness':
            centrality_measure = ga.betweenness_centrality(reuters_pajek)
        elif args.influential_words[1] == 'closeness':
            centrality_measure = ga.closeness_centrality(reuters_pajek)
        else:
            print("Centrality measure doesn't exist.")
            sys.exit(1)

        ga.most_influential_words(args.influential_words[0], centrality_measure)

    if args.word_day_occurrences:
        arguments = args.word_day_occurrences
        if len(arguments) == 1:
            ga.word_day_occurrences(centrality_measure=arguments[0])
        else:
            ga.word_day_occurrences(centrality_file_path=arguments[0],
                                    centrality_measure=arguments[1])

    if args.predict_days:
        features, labels = load_dataset(DATASET_PATH)
        predict_days(features, labels)
