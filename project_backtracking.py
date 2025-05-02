import random
import matplotlib.pyplot as plt
from scipy.spatial import distance
import networkx as nx
import time  # For performance comparison

# Generate a random number of cities on a 2D map
def generate_cities(n):
    cities = [tuple(random.randint(0, 10) for _ in range(2)) for _ in range(n)]  # Grid size 10x10
    return cities

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

# Function to assign colors to cities using the Backtracking Algorithm
def backtracking_coloring(cities, mst_edges):
    # Create a graph representation
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cities)))
    
    # Extract edges correctly from the MST output
    print("MST Edges before adding to graph:", [(edge[0], edge[1]) for edge in mst_edges])
    for edge in mst_edges:
        graph.add_edge(edge[0], edge[1])
    
    # Verify graph structure for debugging
    print("Graph edges after construction:", list(graph.edges()))
    
    # Define available colors with shuffling to encourage variety
    colors = ['Red', 'Blue', 'Green', 'Yellow']
    random.shuffle(colors)  # Shuffle colors to increase variety
    print("Color order being tried:", colors)
    
    # Create a dictionary to track color usage
    color_usage = {color: 0 for color in colors}
    
    # Initialize city colors dictionary with None values
    city_colors = {node: None for node in graph.nodes()}
    
    # Recursive helper function to assign colors with balanced usage
    def color_node(node_idx):
        # Base case: all nodes have been colored
        if node_idx == len(graph.nodes()):
            return True
        
        # Get all neighbors of current node
        neighbors = list(graph.neighbors(node_idx))
        print(f"City {node_idx} has neighbors: {neighbors}")
        
        # Get colors of neighbors
        neighbor_colors = {city_colors[n] for n in neighbors if city_colors[n] is not None}
        print(f"City {node_idx} has neighbor colors: {neighbor_colors}")
        
        # Sort colors by usage (least used first) to encourage variety
        sorted_colors = sorted(colors, key=lambda c: color_usage[c])
        
        # Try each color for the current node
        for color in sorted_colors:
            # Check if this color is valid for the current node
            if color in neighbor_colors:
                continue  # Skip invalid colors
            
            # If valid, assign the color and recurse to the next node
            city_colors[node_idx] = color
            color_usage[color] += 1
            print(f"Assigning {color} to city {node_idx}")
            
            if color_node(node_idx + 1):
                return True
                
            # If assigning this color doesn't lead to a solution,
            # backtrack by trying the next color
            color_usage[color] -= 1
            city_colors[node_idx] = None
            print(f"Backtracking from city {node_idx}")
        
        # If no color works, backtrack to the previous node
        return False
    
    # Start the recursive coloring from node 0
    color_node(0)
    
    # Verify no connected cities share colors
    for u, v in graph.edges():
        if city_colors[u] == city_colors[v]:
            print(f"ERROR: Cities {u} and {v} are connected but both have color {city_colors[u]}")
    
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
    print("Generated cities:", cities)
    
    # Compute the MST
    mst_edges = compute_mst(cities)
    
    # Measure performance
    start_time = time.time()
    
    # Assign colors using the Backtracking Algorithm
    city_colors = backtracking_coloring(cities, mst_edges)
    elapsed_time = time.time() - start_time
    print(f"Backtracking Algorithm execution time: {elapsed_time:.6f} seconds")
    print("City Colors:", city_colors)
    
    # Plot the cities, MST, and colors
    plot_colored_cities_with_mst(cities, mst_edges, city_colors, "Backtracking Algorithm")