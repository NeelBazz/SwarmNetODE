import matplotlib
matplotlib.use('Agg')
import numpy as np
import networkx as nx
import scipy.linalg
import scipy.integrate
import matplotlib.pyplot as plt

N = np.random.randint(10, 101)  
m = max(1, np.random.randint(1, min(N, 5)))  
print(f"Network size: {N}, m: {m}")

G = nx.barabasi_albert_graph(N, m) 

# Compute the Laplacian matrix
L = nx.laplacian_matrix(G).toarray()

# Compute eigenvalues of Laplacian
eigenvalues = np.linalg.eigvalsh(L)
lambda_2 = eigenvalues[1] 

alpha = np.random.uniform(0, 5, N)
beta = np.random.uniform(0, 5, N)
beta_max = np.max(beta)

sigma_min = 4 * beta_max / lambda_2
sigma = np.random.uniform(sigma_min, 2 * sigma_min)  

print(f"Lambda_2: {lambda_2:.4f}, Beta_max: {beta_max:.4f}, Sigma: {sigma:.4f}")

# Step 6: Define the differential equation system
def system_dynamics(t, y):
    x = y[:N]  
    v = y[N:]  

    dxdt = v
    dvdt = -alpha * v - 4 * beta * (x**3 - x) - sigma * np.dot(L, x)
    
    return np.concatenate((dxdt, dvdt))

# Initial conditions: Random small values
x0 = np.random.uniform(-1, 1, N)
v0 = np.random.uniform(-1, 1, N)
y0 = np.concatenate((x0, v0))

t_span = (0, 10) 
t_eval = np.linspace(t_span[0], t_span[1], 1000) 

# Solve the system using Runge-Kutta method
sol = scipy.integrate.solve_ivp(system_dynamics, t_span, y0, t_eval=t_eval, method='RK45')

max_x = np.max(np.abs(sol.y[:N, :]))

print(f"Max |x_i| over time: {max_x:.4f}")

# Monte Carlo Simulation 
num_simulations = 100
max_x_values = []

for _ in range(num_simulations):
    # New initial conditions for each run
    x0 = np.random.uniform(-1, 1, N)
    v0 = np.random.uniform(-1, 1, N)
    y0 = np.concatenate((x0, v0))

    # Solve again
    sol = scipy.integrate.solve_ivp(system_dynamics, t_span, y0, t_eval=t_eval, method='RK45')
    max_x_values.append(np.max(np.abs(sol.y[:N, :])))

# Plot distribution of max |x_i| values
plt.hist(max_x_values, bins=20, alpha=0.7, color='b', edgecolor='black')
plt.xlabel("Max |x_i|")
plt.ylabel("Frequency")
plt.title("Monte Carlo Simulation of Max |x_i|")
plt.show()

plt.savefig("monte_carlo_histogram.png")
print("Plot saved as monte_carlo_histogram.png")

plt.figure(figsize=(8,6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
plt.title(f"Barabási–Albert Graph with {N} Nodes, m={m}")
plt.savefig("barabasi_albert_graph.png") 
print("Graph plot saved as barabasi_albert_graph.png")
