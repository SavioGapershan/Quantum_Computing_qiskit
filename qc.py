import networkx as nx
import matplotlib.pyplot as plt
from qiskit_aer import Aer
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import QAOA
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit.circuit.library import RealAmplitudes,QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import Maxcut
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA



# Step 1: Define the problem graph with added complexity
graph = nx.Graph()
edges = [
    (0, 1, 3), (0, 2, 5), (0, 3, 6), (1, 4, 2),
    (1, 5, 3), (2, 6, 4), (3, 7, 5), (4, 5, 7),
    (5, 6, 1), (6, 7, 8), (4, 7, 2), (2, 5, 3)
]
graph.add_weighted_edges_from(edges)

# Step 2: Visualize the problem graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_size=800, node_color='lightblue')
labels = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
plt.title("Problem Graph: Logistics Centers")
plt.show()

# Step 3: Convert the problem to QUBO format using Max-Cut (Ising model)
num_nodes = len(graph.nodes)
w = nx.adjacency_matrix(graph).todense()
qp = QuadraticProgram()

# Define binary variables
for i in range(num_nodes):
    qp.binary_var(name=f'x{i}')

# Define quadratic and linear parts of the problem (Max-Cut formulation)
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        if w[i, j] != 0:
            qp.minimize(linear={f'x{i}': 0, f'x{j}': 0},
                        quadratic={(f'x{i}', f'x{j}'): -w[i, j]})


# Step 5: Define the QAOA circuit , Create the Max-Cut Hamiltonian (QUBO to Pauli representation)
maxcut = Maxcut(graph)
hamiltonian = maxcut.to_quadratic_program()

hamiltonian = hamiltonian.to_ising();
# Step 6: Use MinimumEigenOptimizer with QAOA
vqe = QAOA(optimizer=COBYLA(), reps=1, sampler=Sampler(),initial_point=[0.95]*len(hamiltonian))
optimizer = MinimumEigenOptimizer(vqe)

# Solve the problem
result = optimizer.solve(qp)

# Step 7: Extract and visualize the solution
solution = result.x
cut_value = result.fval

# Visualize the solution graph
colors = ['red' if solution[i] == 1 else 'blue' for i in range(num_nodes)]
plt.figure(figsize=(8, 6))
nx.draw(graph, pos, with_labels=True, node_size=800, node_color=colors)
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
plt.title(f"Solution Graph: Maximized Cut (Value = {cut_value})")
plt.show()

# Step 8: Visualize the divided graph
group_1 = [i for i in range(num_nodes) if solution[i] == 1]
group_2 = [i for i in range(num_nodes) if solution[i] == 0]

plt.figure(figsize=(8, 6))
nx.draw(graph, pos, with_labels=True, node_size=800, node_color='lightblue', edge_color='gray')
nx.draw_networkx_nodes(graph, pos, nodelist=group_1, node_color='red', label='Group 1')
nx.draw_networkx_nodes(graph, pos, nodelist=group_2, node_color='blue', label='Group 2')
nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), edge_color='gray')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
plt.legend()
plt.title("Divided Graph: Two Groups (Partitioned)")
plt.show()

# Step 10: Display the QAOA quantum circuit with improved label alignment
qaoa_circuit = vqe.ansatz  # Access the QAOAAnsatz directly

# Draw the quantum circuit with improved label alignment
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
qaoa_circuit.draw(output='mpl', ax=ax, style={'gate_color': '#ffcc00', 'label_color': '#0000ff'})  # Customize colors for better visibility

# Set a title for the quantum circuit plot
ax.set_title("QAOA Quantum Circuit")
plt.show()

# Step 9: Output the solution and cost
print("Optimal Cut:", solution)
print("Maximized Cut Value (Cost):", cut_value)
