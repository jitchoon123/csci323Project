import random
import matplotlib.pyplot as plt
from scipy.spatial import distance
import networkx as nx
import time

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

def dsatur_coloring(cities, connections):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cities)))
    for edge in connections:
        graph.add_edge(edge[0], edge[1])
    colors = ['Red', 'Blue', 'Green', 'Yellow']
    color_usage = {color: 0 for color in colors}
    city_colors = {}
    saturation = {node: 0 for node in graph.nodes()}
    def compute_saturation(node):
        neighbor_colors = set()
        for neighbor in graph.neighbors(node):
            if neighbor in city_colors:
                neighbor_colors.add(city_colors[neighbor])
        return len(neighbor_colors)
    while len(city_colors) < len(graph.nodes()):
        max_saturation = -1
        max_node = None
        for node in graph.nodes():
            if node not in city_colors:
                node_saturation = compute_saturation(node)
                if node_saturation > max_saturation or (node_saturation == max_saturation and graph.degree(node) > graph.degree(max_node or 0)):
                    max_saturation = node_saturation
                    max_node = node
        if max_node is None:
            break
        neighbor_colors = set()
        for neighbor in graph.neighbors(max_node):
            if neighbor in city_colors:
                neighbor_colors.add(city_colors[neighbor])
        available_colors = [c for c in colors if c not in neighbor_colors]
        sorted_colors = sorted(available_colors, key=lambda c: color_usage[c])
        if sorted_colors:
            city_colors[max_node] = sorted_colors[0]
            color_usage[sorted_colors[0]] += 1
        for neighbor in graph.neighbors(max_node):
            if neighbor not in city_colors:
                saturation[neighbor] = compute_saturation(neighbor)
    for city in graph.nodes():
        if city not in city_colors:
            available_colors = [c for c in colors]
            city_colors[city] = available_colors[0]
            color_usage[available_colors[0]] += 1
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
    city_colors = dsatur_coloring(cities, connections)
    elapsed_time = time.time() - start_time
    print(f"DSATUR Algorithm execution time: {elapsed_time:.6f} seconds")
    print("City Colors:", city_colors)
    plot_colored_cities_with_connections(cities, connections, city_colors, "DSATUR Algorithm")