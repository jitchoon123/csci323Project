import random
import matplotlib.pyplot as plt
from scipy.spatial import distance
import networkx as nx
import time

# Common Functions
def generate_cities(n):
    """Generate n cities with unique coordinates on a 10x10 grid"""
    cities = set()
    while len(cities) < n:
        city = tuple(random.randint(0, 10) for _ in range(2))
        cities.add(city)
    return list(cities)

## def compute_mst(cities):
    """Compute Minimum Spanning Tree using Kruskal's algorithm"""
    graph = nx.Graph()
    for i, city1 in enumerate(cities):
        for j, city2 in enumerate(cities):
            if i != j:
                dist = distance.euclidean(city1, city2)
                graph.add_edge(i, j, weight=dist)
    
    mst = nx.minimum_spanning_tree(graph)
    return mst.edges(data=True)

def check_intersection(p1, p2, p3, p4):
    """Check if line segments (p1,p2) and (p3,p4) intersect"""
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

def compute_maximal_connections(cities):
    """Compute maximal non-intersecting connections between cities"""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cities)))
    
    # Try all possible connections
    possible_edges = []
    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            possible_edges.append((i, j, distance.euclidean(cities[i], cities[j])))
    
    # Sort edges by distance (shorter edges first)
    possible_edges.sort(key=lambda x: x[2])
    
    # Add edges if they don't intersect with existing ones
    for edge in possible_edges:
        i, j, dist = edge
        can_add = True
        
        # Check intersection with existing edges
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

def plot_colored_cities_with_connections(cities, connections, city_colors, algorithm_name):
    """Plot the cities, connections, and colors with proper legend"""
    x_coords, y_coords = zip(*cities)
    plt.figure(figsize=(10, 10))
    
    # Default color for cities without assigned colors
    default_color = 'Gray'
    color_map = {
        'Red': 'red',
        'Blue': 'blue',
        'Green': 'green',
        'Yellow': 'yellow',
        'Gray': 'gray'  # Add default color
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
        color = color_map[city_colors.get(i, default_color)]  # Use get() with default
        plt.scatter(x, y, c=color, marker='o', s=80, edgecolors='black')
        plt.annotate(str(i), (x, y), fontsize=8, ha='center', va='center')
    
    plt.title(f"2D Map of Cities with Non-Intersecting Connections ({algorithm_name})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend(handles=legend_elements)
    plt.show()

# Algorithm 1: Greedy Coloring
def greedy_coloring(cities, connections):
    """Implement greedy coloring algorithm"""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cities)))
    
    for edge in connections:
        graph.add_edge(edge[0], edge[1])
    
    colors = ['Red', 'Blue', 'Green', 'Yellow']
    color_usage = {color: 0 for color in colors}
    nodes_by_degree = sorted(graph.nodes(), key=lambda n: graph.degree(n), reverse=True)
    city_colors = {}
    
    for city in nodes_by_degree:
        neighbor_colors = set()
        for neighbor in graph.neighbors(city):
            if neighbor in city_colors:
                neighbor_colors.add(city_colors[neighbor])
        
        available_colors = [c for c in colors if c not in neighbor_colors]
        sorted_colors = sorted(available_colors, key=lambda c: color_usage[c])
        
        if sorted_colors:
            city_colors[city] = sorted_colors[0]
            color_usage[sorted_colors[0]] += 1
    
    # Ensure all cities get a color
    for city in graph.nodes():
        if city not in city_colors:
            available_colors = [c for c in colors]
            city_colors[city] = available_colors[0]
            color_usage[available_colors[0]] += 1
    
    return city_colors

# Algorithm 2: DSATUR Coloring
def dsatur_coloring(cities, connections):
    """Implement DSATUR coloring algorithm"""
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
    
    # Ensure all cities get a color
    for city in graph.nodes():
        if city not in city_colors:
            available_colors = [c for c in colors]
            city_colors[city] = available_colors[0]
            color_usage[available_colors[0]] += 1
    
    return city_colors

# Algorithm 3: Backtracking Coloring
def backtracking_coloring(cities, connections):
    """Implement backtracking coloring algorithm"""
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
    
    # Ensure successful coloring
    success = color_node(0)
    if not success:
        # If backtracking fails, assign default colors
        for node in graph.nodes():
            if city_colors[node] is None:
                city_colors[node] = colors[0]
                color_usage[colors[0]] += 1
    
    return city_colors

def main():
    """Main function to run the program"""
    print("\nCity Coloring Algorithms")
    print("1. Greedy Coloring")
    print("2. DSATUR Algorithm")
    print("3. Backtracking Algorithm")
    
    while True:
        try:
            choice = int(input("\nSelect algorithm (1-3): "))
            if choice not in [1, 2, 3]:
                print("Please select a valid option (1-3)")
                continue
            break
        except ValueError:
            print("Please enter a valid number")
    
    while True:
        try:
            num_cities = int(input("Enter the number of cities to generate: "))
            if num_cities <= 0:
                print("Please enter a positive number")
                continue
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Generate cities and compute maximal connections
    cities = generate_cities(num_cities)
    connections = compute_maximal_connections(cities)
    
    # Run selected algorithm
    start_time = time.time()
    
    if choice == 1:
        algorithm_name = "Greedy Algorithm"
        city_colors = greedy_coloring(cities, connections)
    elif choice == 2:
        algorithm_name = "DSATUR Algorithm"
        city_colors = dsatur_coloring(cities, connections)
    else:
        algorithm_name = "Backtracking Algorithm"
        city_colors = backtracking_coloring(cities, connections)
    
    elapsed_time = time.time() - start_time
    print(f"\n{algorithm_name} execution time: {elapsed_time:.6f} seconds")
    print("City Colors:", city_colors)
    
    # Plot results
    plot_colored_cities_with_connections(cities, connections, city_colors, algorithm_name)

if __name__ == "__main__":
    main()