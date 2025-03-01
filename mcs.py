import matplotlib
matplotlib.use('Agg')
import numpy as np
import networkx as nx
import scipy.integrate
import matplotlib.pyplot as plt
import random
import csv

# Parameters
num_graphs = 1000  
num_simulations = 1000  
N = np.random.randint(10, 101)  
m = max(1, np.random.randint(1, min(N, 5)))  

print(f"Generating characteristics for {num_graphs} graphs with N={N}, m={m}")

# Step 1: Store only graph characteristics
graph_characteristics = []

for i in range(num_graphs):
    G = nx.barabasi_albert_graph(N, m)
    L = nx.laplacian_matrix(G).toarray()
    eigenvalues = np.linalg.eigvalsh(L)
    lambda_2 = eigenvalues[1]  
    avg_clustering = nx.average_clustering(G)  # Compute clustering coefficient
    
    graph_characteristics.append({
        "lambda_2": lambda_2,
        "avg_clustering": avg_clustering,
        "edges": G.number_of_edges(),
        "graph_seed": i  # Used to regenerate graphs when needed
    })

# Define a cutoff based on clustering coefficient
clustering_threshold = np.percentile([g["avg_clustering"] for g in graph_characteristics], 75)  # Top 25% most clustered graphs
filtered_graphs = [g for g in graph_characteristics if g["avg_clustering"] >= clustering_threshold]

print(f"Selected {len(filtered_graphs)} graphs for simulation (cutoff Clustering ≥ {clustering_threshold:.4f})")

# Function to simulate dynamics on a given graph
def run_simulation(graph_data):
    # Regenerate the graph using the stored seed
    G = nx.barabasi_albert_graph(N, m, seed=graph_data["graph_seed"])
    L = nx.laplacian_matrix(G).toarray()

    # Compute system parameters
    alpha = np.random.uniform(0, 5, N)
    beta = np.random.uniform(0, 5, N)
    beta_max = np.max(beta)

    sigma_min = 4 * beta_max / graph_data["lambda_2"]
    sigma = np.random.uniform(sigma_min, 2 * sigma_min)  

    # Define system dynamics
    def system_dynamics(t, y):
        x = y[:N]  
        v = y[N:]  
        dxdt = v
        dvdt = -alpha * v - 4 * beta * (x**3 - x) - sigma * np.dot(L, x)
        return np.concatenate((dxdt, dvdt))

    # Initial conditions
    x0 = np.random.uniform(-1, 1, N)
    v0 = np.random.uniform(-1, 1, N)
    y0 = np.concatenate((x0, v0))

    t_span = (0, 10) 
    t_eval = np.linspace(t_span[0], t_span[1], 1000) 

    # Solve system using Runge-Kutta
    sol = scipy.integrate.solve_ivp(system_dynamics, t_span, y0, t_eval=t_eval, method='RK45')

    max_x = np.max(np.abs(sol.y[:N, :]))
    return max_x

# Monte Carlo Simulation with graph sampling from the filtered set
simulation_results = []

for _ in range(num_simulations):
    sampled_graph = random.choice(filtered_graphs)  
    max_x = run_simulation(sampled_graph)

    # Store results
    simulation_results.append({
        "Graph Seed": sampled_graph["graph_seed"],
        "Clustering Coefficient": sampled_graph["avg_clustering"],
        "Lambda_2": sampled_graph["lambda_2"],
        "Max |x_i|": max_x
    })

# Print results
print("\nSimulation Results:")
for result in simulation_results[:10]:  # Display only first 10 results
    print(result)

# Save results to CSV
csv_filename = "simulation_results.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["Graph Seed", "Clustering Coefficient", "Lambda_2", "Max |x_i|"])
    writer.writeheader()
    writer.writerows(simulation_results)

print(f"\nResults saved to {csv_filename}")

# Plot histogram of max |x_i| values
plt.hist([res["Max |x_i|"] for res in simulation_results], bins=30, alpha=0.7, color='b', edgecolor='black')
plt.xlabel("Max |x_i|")
plt.ylabel("Frequency")
plt.title("Monte Carlo Simulation of Max |x_i| (Filtered by Clustering Coefficient)")
plt.savefig("monte_carlo_histogram_filtered_clustering.png")
print("Histogram saved as monte_carlo_histogram_filtered_clustering.png")

# Plot one of the sampled graphs
sampled_graph_data = random.choice(filtered_graphs)
sampled_graph = nx.barabasi_albert_graph(N, m, seed=sampled_graph_data["graph_seed"])

plt.figure(figsize=(8,6))
nx.draw(sampled_graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
plt.title(f"Sampled Barabási–Albert Graph (N={N}, m={m}, Clustering={sampled_graph_data['avg_clustering']:.4f})")
plt.savefig("filtered_sampled_graph_clustering.png")
print("Graph plot saved as filtered_sampled_graph_clustering.png")
