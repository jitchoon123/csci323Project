import random
import matplotlib.pyplot as plt
from scipy.spatial import distance
import networkx as nx

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

# Function to plot the cities and MST
def plot_cities_with_mst(cities, mst_edges):
    x_coords, y_coords = zip(*cities)  # Unpack city coordinates
    plt.figure(figsize=(10, 10))
    
    # Plot the cities
    plt.scatter(x_coords, y_coords, c='blue', marker='o', label='Cities')
    
    # Plot the MST edges
    for edge in mst_edges:
        city1, city2, _ = edge
        x = [cities[city1][0], cities[city2][0]]
        y = [cities[city1][1], cities[city2][1]]
        plt.plot(x, y, c='black', linestyle='-', linewidth=1, label='Roads' if edge == list(mst_edges)[0] else "")
    
    plt.title("2D Map of Cities with MST")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    num_cities = int(input("Enter the number of cities to generate: "))
    cities = generate_cities(num_cities)
    print("Generated cities:", cities)
    
    # Compute the MST
    mst_edges = compute_mst(cities)
    
    # Plot the cities and MST
    plot_cities_with_mst(cities, mst_edges)