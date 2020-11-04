import networkx as net
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import SimRank as sim  #code from graphsim library
import pandas as pd


students = ['Student A','Student B']
courses = ['AAlgorithms','ML','SOE','PBL','Compilers','SWI']

#users = ['User A','User B']
#items = ['Sugar','Frosting','SOE','PBL','Compilers','SWI']


G = net.DiGraph()
G.add_nodes_from(students,bipartite = 0)
G.add_nodes_from(courses,bipartite = 1)
list_nodes = list(G.nodes())
G.add_edge(list_nodes[0],list_nodes[4])
G.add_edge(list_nodes[0],list_nodes[5])
G.add_edge(list_nodes[0],list_nodes[3])
G.add_edge(list_nodes[1],list_nodes[5])
G.add_edge(list_nodes[1],list_nodes[3])
G.add_edge(list_nodes[1],list_nodes[2])
G.add_edge(list_nodes[0],list_nodes[6])
G.add_edge(list_nodes[1],list_nodes[7])


simrbipartite = sim.simrank_bipartite(G)

print(simrbipartite)

pairwise_matrix = net.to_numpy_matrix(G)  #transforms graph to matrix
df = pd.DataFrame(data=simrbipartite, index=G.nodes())  #graph to dataframe
df.columns = G.nodes

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(df)


#print(df.corr('pearson'))   #pearson correlation



# visualization of the graph

fig,ax= plt.subplots()
pos = net.bipartite_layout(G,courses)
net.draw(G,pos,with_labels = True)
net.draw_networkx_labels(G,pos,font_color = 'white')
net.draw_networkx_edges(G, pos, arrowstyle='->',
                        arrowsize=10, edge_color='green',
                        width=2,with_labels=True)
ax.set_facecolor('black')
ax.axis('off')
fig.set_facecolor('black')
plt.show(facecolor="black")

