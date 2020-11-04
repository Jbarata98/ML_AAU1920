import csv as csv
import networkx as net
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import SimRank as sim

def graph_generator(file_name):
    file = open(file_name, 'r')
    graphreader = csv.reader(file, delimiter = '\t')
    G = net.DiGraph()
    for row in graphreader:
        G.add_node(row[0])
    nodes = list(G.nodes())
    print(nodes)
    G.add_edge(nodes[5],nodes[6])
    G.add_edge(nodes[5],nodes[7])
    G.add_edge(nodes[5],nodes[8])
    G.add_edge(nodes[6],nodes[4])
    G.add_edge(nodes[7],nodes[4])
    G.add_edge(nodes[8],nodes[4])
    G.add_edge(nodes[4],nodes[0])
    G.add_edge(nodes[4],nodes[1])
    G.add_edge(nodes[4],nodes[2])
    G.add_edge(nodes[4],nodes[3])
    G.add_edge(nodes[0],nodes[6])
    G.add_edge(nodes[1],nodes[7])
    G.add_edge(nodes[2],nodes[8])
    G.add_edge(nodes[3],nodes[5])

    return G


graph = graph_generator('graph_simrank.csv')

pairwise_matrix = net.to_numpy_matrix(graph)

sim = net.simrank_similarity_numpy(graph)


df = pd.DataFrame(data=sim, index=graph.nodes)
df.columns = graph.nodes


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


#print(df)

pairwise_matrix = net.to_numpy_matrix(graph)

#print(pairwise_matrix)


#cosine_similarity = pairwise_distances(pairwise_matrix, metric="cosine")  #cosine similarity
#print("cosine similarity of student and AAU:" , cosine_similarity[4][5])
#print("cosine similarity of Professor A and AAU:" , cosine_similarity[6][5])


#   graph visualization

fig,ax= plt.subplots()
pos=net.kamada_kawai_layout(graph)
net.draw(graph,pos,with_labels = True)
net.draw_networkx_labels(graph,pos,font_color = 'white')
net.draw_networkx_edges(graph, pos, arrowstyle='->',
                              arrowsize=10, edge_color='green',
                                 width=2)

ax.set_facecolor('black')
fig.set_facecolor('black')
plt.show()

