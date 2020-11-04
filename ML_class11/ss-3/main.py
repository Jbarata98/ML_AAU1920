import numpy as np
import networkx as net
import math
import matplotlib.pyplot as plt

def create_simple_graph():
    G = net.Graph()

    # adding users
    G.add_node('u1')
    G.add_node('u2')
    G.add_node('u3')

    # adding locations
    G.add_node('l1')
    G.add_node('l2')
    G.add_node('l3')

    # adding friendship edges
    G.add_edge('u1', 'u3')
    G.add_edge('u2', 'u3')

    # adding visits edges
    G.add_edge('u1', 'l1', weight=7)
    G.add_edge('u1', 'l2', weight=3)
    G.add_edge('u3', 'l2', weight=1)
    G.add_edge('u3', 'l3', weight=1)

    return G

def BCA(G : net.Graph,
        user,
        epsilon : float = 1e-10,
        alpha : float = 0.85,
        steps : int = 50):
    p = {} # pageranks
    d = {} # friendships
    b = {} # size |U|

    nodes = G.nodes
    # all nodes that are users, i.e. 'ux'
    users = [u for u in list(nodes) if u.startswith('u')]

    # initializing pageranks and friendships
    for i in users:
        p[i] = 0
        # friends are neighbors in graph
        neighbors = G.neighbors(i)
        friends = [u for u in list(neighbors) if u.startswith('u')]
        d[i] = len(list(friends))
        b[i] = 1 if i == user else 0 # b_u = 1, rest is 0
    #print(d)
    #print(b)
    # run iterations
    #print(p)
    for step in range(steps):
        # used to know if our b has converged
        b_prev = b.copy()
        #print(b_prev)
        for i in users:
            # if our threshold has been reached user has nothing to distribute.
            if b[i] < epsilon:
                continue
            # update pageranks.
            p[i] += (1 - alpha) * b[i]
            #print(p)

            # distribute color to neighbors
            neighbors = G.neighbors(i)
            friends = [u for u in list(neighbors) if u.startswith('u')]
            #print(friends)
            for friend in friends:
                b[friend] += alpha * b[i]/d[i]
                #print(b[friend])
                #print(friend)

            # keep (1-alpha) yourself
            b[i] = (1 - alpha) * b[i]

        if b_prev == b:
            break

    # since we increase values of p in each iteration
    # it is not ensured that p sums to 1.
    # Fix could be to simply normalize it.

    return p

def FBCA(G : net.Graph,
         user,
         k : int):
    # locations user has visited
    l_u = [l for l in list(G.neighbors(user)) if l.startswith('l')]
    #print(l_u)

    # all locations
    locations = [l for l in list(G.nodes) if l.startswith('l')]

    # locations user has not visited
    locations_not_visited = [l for l in locations if l not in l_u]

    # different users
    different_users = [u for u in list(G.nodes) if u.startswith('u') and u != user]

    # compute PPR for all users
    PPR = BCA(G, user)

    # initializing scores of locations that user has not visited
    scores = {}
    for l in locations_not_visited:

        scores[l] = 0
    #print(scores)
    for u in different_users:
        print(u)
        for l in locations:
            #print(u)
            locations_visited = [l for l in G.neighbors(u) if l.startswith('l')]
            print(locations_visited)
            if l not in locations_visited: # if user has not visited location skip it
                continue
            n_visits = G.get_edge_data(u, l)['weight']

            scores[l] += PPR[u] * n_visits
            #print(scores)

    # sorting scores
    sorted_scores = sorted(scores.items(), reverse=True, key=lambda kv: kv[1])

    # if k exceeds length of scored locations, recommend all locations instead
    if k < len(sorted_scores):
        k = len(sorted_scores)

    return sorted_scores[:k-1]

if __name__ == '__main__':
    G = create_simple_graph()

    #net.draw(G, with_labels=True)
    #plt.show()

    test_user = 'u2'
    recommendations = FBCA(G, test_user,2)

    print(recommendations)

    # maybe do the simple alg (alg 1)
    # and prepare for how to extend to alg 2 and 3.

    # alternatively implement alg 2 or 3.