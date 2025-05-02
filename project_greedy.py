import random
import matplotlib.pyplot as plt
from scipy.spatial import distance
import networkx as nx
import time

# Fix for generating cities with unique coordinates
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

# Fix for greedy coloring algorithm
def greedy_coloring(cities, mst_edges):
    # Create a graph representation
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cities)))
    
    # Extract edges correctly from the MST output
    for edge in mst_edges:
        graph.add_edge(edge[0], edge[1])
    
    # Define available colors
    colors = ['Red', 'Blue', 'Green', 'Yellow']
    
    # Create a dictionary to track color usage
    color_usage = {color: 0 for color in colors}
    
    # Process cities in degree order (higher degree first)
    nodes_by_degree = sorted(graph.nodes(), key=lambda n: graph.degree(n), reverse=True)
    
    # Initialize city colors dictionary
    city_colors = {}
    
    # Assign colors to each city
    for city in nodes_by_degree:
        # Find colors used by neighboring cities
        neighbor_colors = set()
        for neighbor in graph.neighbors(city):
            if neighbor in city_colors:
                neighbor_colors.add(city_colors[neighbor])
        
        # Sort available colors by usage (least used first)
        available_colors = [c for c in colors if c not in neighbor_colors]
        sorted_colors = sorted(available_colors, key=lambda c: color_usage[c])
        
        if sorted_colors:
            # Assign the least used color that's not used by neighbors
            city_colors[city] = sorted_colors[0]
            color_usage[sorted_colors[0]] += 1
        else:
            print(f"ERROR: No available color for city {city}")
            # This should never happen with 4 colors and a tree structure
            # But if it does, backtrack or use a different algorithm
            
    # Double-check for violations
    violations = False
    for u, v in graph.edges():
        if city_colors[u] == city_colors[v]:
            print(f"ERROR: Cities {u} and {v} are connected but both have color {city_colors[u]}")
            violations = True
    
    if violations:
        print("WARNING: Color constraints violated. Please check the coloring algorithm.")
    
    return city_colors

# Improved plotting function
def plot_colored_cities_with_mst(cities, mst_edges, city_colors):
    x_coords, y_coords = zip(*cities)  # Unpack city coordinates
    plt.figure(figsize=(10, 10))
    
    # Define a color mapping for matplotlib
    color_map = {
        'Red': 'red',
        'Blue': 'blue',
        'Green': 'green',
        'Yellow': 'yellow'
    }
    
    # Create a legend mapping for colors
    legend_elements = []
    for color in color_map:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=color_map[color], markersize=10, label=color))
    
    # Plot the MST edges first (so they appear behind the cities)
    for edge in mst_edges:
        city1, city2, _ = edge
        x = [cities[city1][0], cities[city2][0]]
        y = [cities[city1][1], cities[city2][1]]
        plt.plot(x, y, c='black', linestyle='-', linewidth=1)
    
    # Plot the cities with their assigned colors (make markers smaller)
    for i, (x, y) in enumerate(cities):
        plt.scatter(x, y, c=color_map[city_colors[i]], marker='o', s=80, edgecolors='black')
        plt.annotate(str(i), (x, y), fontsize=8, ha='center', va='center')
    
    plt.title("2D Map of Cities with MST and Colors")
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
    
    # Assign colors using the Greedy Coloring Algorithm
    city_colors = greedy_coloring(cities, mst_edges)
    elapsed_time = time.time() - start_time
    print(f"Greedy Colour execution time: {elapsed_time:.6f} seconds")
    
    # Plot the cities, MST, and colors
    plot_colored_cities_with_mst(cities, mst_edges, city_colors)