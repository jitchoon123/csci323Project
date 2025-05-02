import random
import matplotlib.pyplot as plt
from scipy.spatial import distance
import networkx as nx
import time  # For performance comparison

# Generate a random number of cities on a 2D map
def generate_cities(n):
    cities = set()
    while len(cities) < n:
        city = tuple(random.randint(0, 10) for _ in range(2))
        cities.add(city)
    return list(cities)

# Function to compute the MST to connect all cities
def compute_mst(cities):
    # Create a complete graph with distances as weights
    graph = nx.Graph()
    for i, city1 in enumerate(cities):
        for j, city2 in enumerate(cities):
            if i != j:
                dist = distance.euclidean(city1, city2)
                graph.add_edge(i, j, weight=dist)
    
    # Compute the MST using Kruskal's algorithm
    mst = nx.minimum_spanning_tree(graph)
    return mst.edges(data=True)

# Function to assign colors to cities using the DSATUR Algorithm
def dsatur_coloring(cities, mst_edges):
    # Create a graph representation
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cities)))
    
    # Extract edges correctly from the MST output
    for edge in mst_edges:
        graph.add_edge(edge[0], edge[1])
    
    # Define available colors with intentional order to encourage variety
    colors = ['Red', 'Blue', 'Green', 'Yellow']
    
    # Create a dictionary to track color usage
    color_usage = {color: 0 for color in colors}
    
    # Initialize city colors dictionary
    city_colors = {}
    
    # Track the saturation degree (number of different colors in the neighborhood)
    saturation = {node: 0 for node in graph.nodes()}
    
    # Function to compute current saturation of a node
    def compute_saturation(node):
        neighbor_colors = set()
        for neighbor in graph.neighbors(node):
            if neighbor in city_colors:
                neighbor_colors.add(city_colors[neighbor])
        return len(neighbor_colors)
    
    # While not all nodes are colored
    while len(city_colors) < len(graph.nodes()):
        # Find the uncolored node with the highest saturation
        max_saturation = -1
        max_node = None
        
        for node in graph.nodes():
            if node not in city_colors:  # Uncolored node
                node_saturation = compute_saturation(node)
                if node_saturation > max_saturation or (node_saturation == max_saturation and graph.degree(node) > graph.degree(max_node or 0)):
                    max_saturation = node_saturation
                    max_node = node
        
        # If all nodes are colored, break
        if max_node is None:
            break
        
        # Find colors used by neighbors
        neighbor_colors = set()
        for neighbor in graph.neighbors(max_node):
            if neighbor in city_colors:
                neighbor_colors.add(city_colors[neighbor])
        
        # Sort available colors by usage (least used first)
        available_colors = [c for c in colors if c not in neighbor_colors]
        sorted_colors = sorted(available_colors, key=lambda c: color_usage[c])
        
        if sorted_colors:
            # Assign the least used color that's not used by neighbors
            city_colors[max_node] = sorted_colors[0]
            color_usage[sorted_colors[0]] += 1
        
        # Update saturation of uncolored neighbors
        for neighbor in graph.neighbors(max_node):
            if neighbor not in city_colors:
                saturation[neighbor] = compute_saturation(neighbor)
    
    # Verify no connected cities share colors
    violations = False
    for u, v in graph.edges():
        if city_colors[u] == city_colors[v]:
            violations = True
    
    return city_colors

# Improved plotting function
def plot_colored_cities_with_mst(cities, mst_edges, city_colors, algorithm_name):
    x_coords, y_coords = zip(*cities)  # Unpack city coordinates
    plt.figure(figsize=(10, 10))
    
    # Define a color mapping for matplotlib
    color_map = {
        'Red': 'red',
        'Blue': 'blue',
        'Green': 'green',
        'Yellow': 'yellow'
    }
    
    # Create proper legend elements for colors
    legend_elements = []
    for color in color_map:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=color_map[color], markersize=10, label=color))
    
    # Add a legend element for roads
    legend_elements.append(plt.Line2D([0], [0], color='black', lw=1, label='Roads'))
    
    # Plot the MST edges first (so they appear behind the cities)
    for edge in mst_edges:
        city1, city2, _ = edge
        x = [cities[city1][0], cities[city2][0]]
        y = [cities[city1][1], cities[city2][1]]
        plt.plot(x, y, c='black', linestyle='-', linewidth=1)
    
    # Plot the cities with their assigned colors
    for i, (x, y) in enumerate(cities):
        plt.scatter(x, y, c=color_map[city_colors[i]], marker='o', s=80, edgecolors='black')
        # Add city number labels
        plt.annotate(str(i), (x, y), fontsize=8, ha='center', va='center')
    
    plt.title(f"2D Map of Cities with MST and Colors ({algorithm_name})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend(handles=legend_elements)
    plt.show()

# Example usage
if __name__ == "__main__":
    num_cities = int(input("Enter the number of cities to generate: "))
    cities = generate_cities(num_cities)
    
    # Compute the MST
    mst_edges = compute_mst(cities)
    
    # Measure performance
    start_time = time.time()
    
    # Assign colors using the DSATUR Algorithm
    city_colors = dsatur_coloring(cities, mst_edges)
    
    elapsed_time = time.time() - start_time
    print(f"DSATUR Algorithm execution time: {elapsed_time:.6f} seconds")
    print("City Colors:", city_colors)
    
    # Plot the cities, MST, and colors
    plot_colored_cities_with_mst(cities, mst_edges, city_colors, "DSATUR Algorithm")