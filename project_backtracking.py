import random
import matplotlib.pyplot as plt
from scipy.spatial import distance
import networkx as nx
import time  # For performance comparison

def generate_cities(n):
    cities = set()
    while len(cities) < n:
        city = tuple(random.randint(0, 10) for _ in range(2))
        cities.add(city)
    return list(cities)

def check_intersection(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

def compute_maximal_connections(cities):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cities)))
    possible_edges = []
    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            possible_edges.append((i, j, distance.euclidean(cities[i], cities[j])))
    possible_edges.sort(key=lambda x: x[2])
    for edge in possible_edges:
        i, j, dist = edge
        can_add = True
        for existing_edge in graph.edges():
            if i in existing_edge or j in existing_edge:
                continue
            if check_intersection(
                cities[i], cities[j],
                cities[existing_edge[0]], cities[existing_edge[1]]
            ):
                can_add = False
                break
        if can_add:
            graph.add_edge(i, j, weight=dist)
    return list(graph.edges(data=True))

def backtracking_coloring(cities, connections):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cities)))
    for edge in connections:
        graph.add_edge(edge[0], edge[1])
    colors = ['Red', 'Blue', 'Green', 'Yellow']
    random.shuffle(colors)
    color_usage = {color: 0 for color in colors}
    city_colors = {node: None for node in graph.nodes()}
    def color_node(node_idx):
        if node_idx == len(graph.nodes()):
            return True
        neighbors = list(graph.neighbors(node_idx))
        neighbor_colors = {city_colors[n] for n in neighbors if city_colors[n] is not None}
        sorted_colors = sorted(colors, key=lambda c: color_usage[c])
        for color in sorted_colors:
            if color in neighbor_colors:
                continue
            city_colors[node_idx] = color
            color_usage[color] += 1
            if color_node(node_idx + 1):
                return True
            color_usage[color] -= 1
            city_colors[node_idx] = None
        return False
    color_node(0)
    return city_colors

def plot_colored_cities_with_connections(cities, connections, city_colors, algorithm_name):
    x_coords, y_coords = zip(*cities)
    plt.figure(figsize=(10, 10))
    color_map = {
        'Red': 'red',
        'Blue': 'blue',
        'Green': 'green',
        'Yellow': 'yellow'
    }
    legend_elements = []
    for color in color_map:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=color_map[color], markersize=10, label=color))
    legend_elements.append(plt.Line2D([0], [0], color='black', lw=1, label='Connections'))
    for edge in connections:
        city1, city2, _ = edge
        x = [cities[city1][0], cities[city2][0]]
        y = [cities[city1][1], cities[city2][1]]
        plt.plot(x, y, c='black', linestyle='-', linewidth=1)
    for i, (x, y) in enumerate(cities):
        plt.scatter(x, y, c=color_map[city_colors[i]], marker='o', s=80, edgecolors='black')
        plt.annotate(str(i), (x, y), fontsize=8, ha='center', va='center')
    plt.title(f"2D Map of Cities with Non-Intersecting Connections ({algorithm_name})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend(handles=legend_elements)
    plt.show()

if __name__ == "__main__":
    num_cities = int(input("Enter the number of cities to generate: "))
    cities = generate_cities(num_cities)
    connections = compute_maximal_connections(cities)
    start_time = time.time()
    city_colors = backtracking_coloring(cities, connections)
    elapsed_time = time.time() - start_time
    print(f"Backtracking Algorithm execution time: {elapsed_time:.6f} seconds")
    print("City Colors:", city_colors)
    plot_colored_cities_with_connections(cities, connections, city_colors, "Backtracking Algorithm")