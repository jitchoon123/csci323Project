import random
import matplotlib.pyplot as plt
from scipy.spatial import distance
import networkx as nx
import time

# ==== COMMON FUNCTIONS ====

def generate_cities(n):
    cities = set()
    while len(cities) < n:
        cities.add(tuple(random.randint(0, 10) for _ in range(2)))
    return list(cities)

def compute_mst(cities):
    graph = nx.Graph()
    for i, city1 in enumerate(cities):
        for j, city2 in enumerate(cities):
            if i != j:
                dist = distance.euclidean(city1, city2)
                graph.add_edge(i, j, weight=dist)
    return nx.minimum_spanning_tree(graph).edges(data=True)

def plot_colored_cities_with_mst(cities, mst_edges, city_colors, algorithm_name):
    x_coords, y_coords = zip(*cities)
    plt.figure(figsize=(10, 10))
    color_map = {'Red': 'red', 'Blue': 'blue', 'Green': 'green', 'Yellow': 'yellow'}
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[c], markersize=10, label=c) for c in color_map]
    legend_elements.append(plt.Line2D([0], [0], color='black', lw=1, label='Roads'))

    for edge in mst_edges:
        city1, city2, _ = edge
        plt.plot([cities[city1][0], cities[city2][0]], [cities[city1][1], cities[city2][1]], c='black', lw=1)

    for i, (x, y) in enumerate(cities):
        plt.scatter(x, y, c=color_map[city_colors[i]], s=80, edgecolors='black')
        plt.annotate(str(i), (x, y), fontsize=8, ha='center', va='center')

    plt.title(f"2D Map of Cities with MST and Colors ({algorithm_name})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend(handles=legend_elements)
    plt.show()

# ==== ALGORITHM IMPLEMENTATIONS ====

def dsatur_coloring(cities, mst_edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cities)))
    for edge in mst_edges:
        graph.add_edge(edge[0], edge[1])

    colors = ['Red', 'Blue', 'Green', 'Yellow']
    color_usage = {color: 0 for color in colors}
    city_colors = {}
    saturation = {node: 0 for node in graph.nodes()}

    def compute_saturation(node):
        return len({city_colors[neighbor] for neighbor in graph.neighbors(node) if neighbor in city_colors})

    while len(city_colors) < len(graph.nodes()):
        max_node = max((node for node in graph.nodes() if node not in city_colors), key=lambda n: (compute_saturation(n), graph.degree(n)))
        neighbor_colors = {city_colors[n] for n in graph.neighbors(max_node) if n in city_colors}
        available_colors = [c for c in colors if c not in neighbor_colors]
        chosen_color = min(available_colors, key=lambda c: color_usage[c])
        city_colors[max_node] = chosen_color
        color_usage[chosen_color] += 1

    return city_colors

def backtracking_coloring(cities, mst_edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cities)))
    for edge in mst_edges:
        graph.add_edge(edge[0], edge[1])

    colors = ['Red', 'Blue', 'Green', 'Yellow']
    random.shuffle(colors)
    city_colors = {node: None for node in graph.nodes()}
    color_usage = {color: 0 for color in colors}

    def color_node(node_idx):
        if node_idx == len(graph.nodes()):
            return True
        neighbors = list(graph.neighbors(node_idx))
        neighbor_colors = {city_colors[n] for n in neighbors if city_colors[n]}
        sorted_colors = sorted(colors, key=lambda c: color_usage[c])

        for color in sorted_colors:
            if color not in neighbor_colors:
                city_colors[node_idx] = color
                color_usage[color] += 1
                if color_node(node_idx + 1):
                    return True
                color_usage[color] -= 1
                city_colors[node_idx] = None
        return False

    color_node(0)
    return city_colors

def greedy_coloring(cities, mst_edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cities)))
    for edge in mst_edges:
        graph.add_edge(edge[0], edge[1])

    colors = ['Red', 'Blue', 'Green', 'Yellow']
    city_colors = {}

    for node in sorted(graph.nodes(), key=lambda n: graph.degree(n), reverse=True):
        used = {city_colors[n] for n in graph.neighbors(node) if n in city_colors}
        for color in colors:
            if color not in used:
                city_colors[node] = color
                break

    return city_colors

# ==== MAIN ENTRY ====

if __name__ == "__main__":
    algorithms = {
        "1": ("DSATUR", dsatur_coloring),
        "2": ("Backtracking", backtracking_coloring),
        "3": ("Greedy", greedy_coloring)
    }

    print("Select algorithm:")
    for k, (name, _) in algorithms.items():
        print(f"{k}: {name}")
    algo_choice = input("Enter choice (1/2/3): ").strip()
    
    if algo_choice not in algorithms:
        print("Invalid choice.")
        exit()

    num_cities = int(input("Enter the number of cities to generate: "))
    cities = generate_cities(num_cities)
    mst_edges = compute_mst(cities)

    name, algo_func = algorithms[algo_choice]
    start_time = time.time()
    city_colors = algo_func(cities, mst_edges)
    elapsed = time.time() - start_time

    print(f"{name} execution time: {elapsed:.4f} seconds")
    print("City Colors:", city_colors)
    plot_colored_cities_with_mst(cities, mst_edges, city_colors, name)
    print("Plotting completed.")