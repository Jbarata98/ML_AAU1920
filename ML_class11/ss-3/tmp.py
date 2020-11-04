def create_example_graph_alg1(n_users: int):
    G = net.Graph()

    # adding users
    for u in range(0, n_users):
        G.add_node(u)

    # adding friendships
    for u1 in G.nodes:
        for u2 in G.nodes:
            random = np.random.random()
            if u1 != u2 and random > 0.5:
                G.add_edge(u1, u2) # simulates friendship

    return G

def create_example_graph_alg2(n_users: int, n_locations: int):
    G = net.Graph()

    # adding user nodes
    for u in range(0, n_users):
        user = "u{0}".format(u+1)
        G.add_node(user)

    # adding location nodes
    for l in range(0, n_locations):
        location = "l{0}".format(l+1)
        G.add_node(location)

    # adding edges "randomly"
    add_edges_randomly(G)

def add_edges_randomly(G: net.Graph):
    for node in G.nodes:
        for neighbor in G.nodes:
            if node == neighbor: # no edge to yourself
                continue
            else:
                node_is_user = str.startswith(node, 'u')
                node_is_location = str.startswith(node, 'l')
                neighbor_is_user = str.startswith(neighbor, 'u')
                neighbor_is_location = str.startswith(neighbor, 'l')

                random = np.random.random() # float between 0-1,
                                            # used to determine whether or not to add edge
                if random > 0.5:
                    if node_is_user and neighbor_is_user:
                        add_friendship_edge(G, node, neighbor)
                    elif node_is_location and neighbor_is_user:
                        visits = math.floor(random * 10)
                        add_visits_edge(G, neighbor, node, weight=visits)
                    elif node_is_user and neighbor_is_location:
                        visits = math.floor(random * 10)
                        add_visits_edge(G, node, neighbor, weight=visits)

def add_visits_edge(graph, node_from, node_to, visits):
    graph.add_edge(node_from, node_to, weight=visits)

def add_friendship_edge(graph, node_from, node_to):
    graph.add_edge(node_from, node_to)
