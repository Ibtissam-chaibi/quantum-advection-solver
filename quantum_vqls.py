import pennylane as qml
from pennylane import numpy as np

def ansatz(theta, wires):
    """Hardware-efficient ansatz with trainable rotations"""
    n_qubits = len(wires)
    layers = len(theta) // (2 * n_qubits)
    idx = 0
    
    for _ in range(layers):
        # First set of rotations
        for i in wires:
            qml.RY(theta[idx], wires=i)
            idx += 1
        
        # Entanglement layer
        for i in range(n_qubits-1):
            qml.CNOT(wires=[wires[i], wires[i+1]])
        
        # Second set of rotations
        for i in wires:
            qml.RY(theta[idx], wires=i)
            idx += 1
        
        # Reverse entanglement
        for i in reversed(range(n_qubits-1)):
            qml.CNOT(wires=[wires[i], wires[i+1]])

def solve_linear_system(A, b, num_layers=1, max_iter=30):
    """VQLS solver for Au = b"""
    n_qubits = int(np.log2(len(b)))
    dev = qml.device("default.qubit", wires=n_qubits)
    num_params = 2 * n_qubits * num_layers
    
    @qml.qnode(dev)
    def circuit(theta):
        ansatz(theta, wires=range(n_qubits))
        return qml.state()
    
    def cost(theta):
        psi = circuit(theta)
        A_psi = A @ psi
        return np.linalg.norm(A_psi - b)**2
    
    # Initialize parameters
    theta = np.random.uniform(0, 2*np.pi, num_params, requires_grad=True)
    opt = qml.AdamOptimizer(0.05)
    
    # Optimization loop
    for i in range(max_iter):
        theta, loss = opt.step_and_cost(cost, theta)
        if i % 5 == 0:
            print(f"Iter {i:3d}: Loss = {loss:.6f}")
    
    # Compute solution
    psi_opt = circuit(theta)
    scale = np.dot(b, A @ psi_opt)
    return np.real(psi_opt) / scale
