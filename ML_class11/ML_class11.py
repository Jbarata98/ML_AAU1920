import csv as csv
import networkx as net
import matplotlib.pyplot as plt
import itertools


def file_open(file_name):   #create_graph
    file = open(file_name, 'r')
    graphreader = csv.reader(file, delimiter = '\t')
    G = net.Graph()
    for row in graphreader:
        G.add_node(row[0],type ='user')
        G.add_node(row[1],type = 'location')
    nodes = list(G.nodes())
    G.add_edge(nodes[0],nodes[1],type='user-visit', weight = 7)
    G.add_edge(nodes[0],nodes[3],type='user-visit', weight = 3)
    G.add_edge(nodes[0],nodes[4],type='user-friendship', weight = 0)
    G.add_edge(nodes[2],nodes[4],type='user-friendship', weight = 0)
    G.add_edge(nodes[4],nodes[3],type='user-visit', weight = 1)
    G.add_edge(nodes[4],nodes[5],type='user-visit', weight = 1)
    G.add_edge(nodes[4],nodes[0],type='user-friendship',weight = 0)

    return G


graph = file_open('graph.csv')

a = 0.85
t_threshold= 1e-8
d_treshold = 2
b = 0.8

list_users = []
list_locations = []
def get_users(graph):
    nodes = list(graph.nodes(data=True))
    for  i in range(len(nodes)):
        if nodes[i][1].get('type') == 'user':
            list_users.append(nodes[i])
    return list_users

def get_locations(graph):

    nodes = list(graph.nodes(data=True))
    for  i in range(len(nodes)):
        if nodes[i][1].get('type') == 'location':
            list_locations.append(nodes[i])
    return list_locations

list_nodes = get_users(graph)

def BCA(graph,u,alpha,t_thresh,iterations): #it has errors
      # all users
    ppr = [0] * len(list_nodes)
    b = [0] * len(list_nodes)
    d = [0] * len(list_nodes)
    friends_list = []
    b[u] = 1
    for i in range(len(list_nodes)):
        ppr[i] = 0
        friends = 0
        x = dict(graph[list_nodes[i][0]])
        for item, value in zip(x.keys(), x.values()):
            if value.get('type') == 'user-friendship':
                friends += 1
                friends_list.append(item)
        d[i] = friends
    for iteration in range(iterations):
        b_prev = b.copy()
        for j in range(len(list_nodes)):
            if b[j] < t_thresh:
                continue
            ppr[j]+= (1-alpha)*b[j]
            x = dict(graph[list_nodes[j][0]])
            friends_list = []
            for item, value in zip(x.keys(), x.values()):
                if value.get('type') == 'user-friendship':
                    friends_list.append(item)
            for friend in friends_list:
                b[int(friend[1])-1]+= alpha*b[j]/(d[j])
            b[j] = (1 - alpha) * b[j]
        if b_prev == b:
            break
    return ppr

def FBCA(graph,u,nr_locations):

    x = dict(graph[list_users[u][0]])
    locations_user = []
    for item, value in zip(x.keys(), x.values()):
        if value.get('type') == 'user-visit':
            locations_user.append(item)

    list_locations = get_locations(graph)


    list_locations_unvisit = [l for l in list_locations if l[0] not in locations_user]
    PPR = BCA(graph, u, a, t_threshold, 50)

    list_users.pop(u)
    list_diff_users = list_users


    scores = d = [0] * len(list_locations)
    for i in list_locations_unvisit:
        scores[int(i[0][1])-1] = 0 #should use dictionary next time
    #print(scores)
    #print(different_users)

    for j in list_diff_users:
        print(j)
        x = graph[j[0]]
        #print(j)
        locations_diff_user = [l for l in graph.neighbors(j[0]) if l.startswith('l')]
        print(locations_diff_user)
        for l in locations_diff_user:
           # n_visits = graph.edges[0][1]['a']['weight']

            scores[int(l[1])-1] += PPR[u] + graph.get_edge_data(str(j[0]),str(l))['weight']
            #print(scores)

    return scores

#print(graph.get_edge_data('u2','l2')['weight'])
u = 2
recommendations = FBCA(graph, u,2)

print(recommendations)
#net.draw(graph,pos=net.spring_layout(graph),node_color ='blue', with_labels=True,font_color = 'white')
#plt.show()