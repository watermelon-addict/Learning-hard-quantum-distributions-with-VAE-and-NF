# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 20:52:00 2023

https://pennylane.ai/qml/demos/qsim_beyond_classical/

@author: Ellen Wang
"""

import pennylane as qml
from pennylane_cirq import ops

import cirq
import numpy as np

import matplotlib.pyplot as plt
import json

# Change the size of the below array to generate quantum distributions with different number of qubits
qubits = sorted([
    cirq.GridQubit(3, 3),
    cirq.GridQubit(3, 4),
    cirq.GridQubit(3, 5),
    cirq.GridQubit(3, 6),
    cirq.GridQubit(4, 3),
    cirq.GridQubit(4, 4),
    cirq.GridQubit(4, 5),
    cirq.GridQubit(4, 6),
    cirq.GridQubit(5, 3),
    cirq.GridQubit(5, 4),
    cirq.GridQubit(5, 5),
    cirq.GridQubit(5, 6),
])

wires = len(qubits)

# create a mapping between wire number and Cirq qubit
qb2wire = {i: j for i, j in zip(qubits, range(wires))}

shots = 5000000
dev = qml.device('cirq.qsim', wires=wires, qubits=qubits, shots=shots)

sqrtYgate = lambda wires: qml.RY(np.pi / 2, wires=wires)

sqrtWgate = lambda wires: qml.QubitUnitary(
    np.array([[1,  -np.sqrt(1j)],
              [np.sqrt(-1j), 1]]) / np.sqrt(2), wires=wires
)

single_qubit_gates = [qml.SX, sqrtYgate, sqrtWgate]

from itertools import combinations

gate_order = {"A":[], "B":[], "C":[], "D":[]}
for i, j in combinations(qubits, 2):
    wire_1 = qb2wire[i]
    wire_2 = qb2wire[j]
    if i in j.neighbors():
        if i.row == j.row and i.col % 2 == 0:
            gate_order["A"].append((wire_1, wire_2))
        elif i.row == j.row and j.col % 2 == 0:
            gate_order["B"].append((wire_1, wire_2))
        elif i.col == j.col and i.row % 2 == 0:
            gate_order["C"].append((wire_1, wire_2))
        elif i.col == j.col and j.row % 2 == 0:
            gate_order["D"].append((wire_1, wire_2))

m = 14  # number of cycles

gate_sequence_longer = np.resize(["A", "B", "C", "D", "C", "D", "A", "B"], m)
gate_sequence = np.resize(["A", "B", "C", "D"], m)

def generate_single_qubit_gate_list():
    # create the first list by randomly selecting indices
    # from single_qubit_gates
    g = [list(np.random.choice(range(len(single_qubit_gates)), size=wires))]

    for cycle in range(len(gate_sequence)):
        g.append([])
        for w in range(wires):
            # check which gate was applied to the wire previously
            one_gate_removed = list(range(len(single_qubit_gates)))
            bool_list = np.array(one_gate_removed) == g[cycle][w]

            # and remove it from the choices of gates to be applied
            pop_idx = np.where(bool_list)[0][0]
            one_gate_removed.pop(pop_idx)
            g[cycle + 1].append(np.random.choice(one_gate_removed))
    return g

@qml.qnode(dev)
def circuit(seed=42, return_probs=False):
    np.random.seed(seed)
    gate_idx = generate_single_qubit_gate_list()

    # m full cycles - single-qubit gates & two-qubit gate
    for i, gs in enumerate(gate_sequence):
        for w in range(wires):
            single_qubit_gates[gate_idx[i][w]](wires=w)

        for qb_1, qb_2 in gate_order[gs]:

            # qml.ISWAP(wires=(qb_1, qb_2))
            # qml.CZ(wires=(qb_1, qb_2))
            qml.SISWAP(wires=(qb_1, qb_2))

            qml.CPhase(-np.pi/6, wires=(qb_1, qb_2))

    # one half-cycle - single-qubit gates only
    for w in range(wires):
        single_qubit_gates[gate_idx[-1][w]](wires=w)

    if return_probs:
        return qml.probs(wires=range(wires))
    else:
        return qml.sample()
    
def fidelity_xeb(samples, probs):
    sampled_probs = []
    for bitstring in samples:
        # convert each bitstring into an integer
        bitstring_idx = int(bitstring, 2)

        # retrieve the corresponding probability for the bitstring
        sampled_probs.append(probs[bitstring_idx])

    return 2 ** len(samples[0]) * np.mean(sampled_probs) - 1

seed = np.random.randint(0, 42424242)
probs = circuit(seed=seed, return_probs=True)
circuit_samples = circuit(seed=seed)

# get bitstrings from the samples
bitstring_samples = []
for sam in circuit_samples:
    bitstring_samples.append("".join(str(bs) for bs in sam))

f_circuit = fidelity_xeb(bitstring_samples, probs)


# save bitstring results as integer numbers
N = wires
saved_file = "Quantum_Distr_qsim_{}bit.json".format(N)
# saved_file = "QSIM_Binary_Data_{}bit_CZ.json".format(N)
def BinStr2Num( Y, N ):
    num = 0
    for i, b in enumerate(Y):
        num += b * 2 **(N-1-i)
    return int(num)

y_out = []
for c in circuit_samples:
    y_out.append( BinStr2Num(c,N))

np.random.shuffle(y_out)
# show histogram of Y distribution
plt.hist(y_out, bins=2**N, range = (0,2**N-1) )
plt.show()

# y_out = [ x.tolist() for x in y_out ]
# y_out = y_out.tolist()
with open( saved_file, "w") as outfile:
    data = {"data": y_out}
    json.dump(data, outfile, indent = 4 )

'''
basis_states = dev.generate_basis_states(wires)
random_integers = np.random.randint(0, len(basis_states), size=shots)
bitstring_samples = []
for i in random_integers:
    bitstring_samples.append("".join(str(bs) for bs in basis_states[i]))

f_uniform = fidelity_xeb(bitstring_samples, probs)

print("Circuit's distribution:", f"{f_circuit:.7f}".rjust(12))
print("Uniform distribution:", f"{f_uniform:.7f}".rjust(14))


N = 2 ** wires
theoretical_value = 2 * N / (N + 1) - 1

print("Theoretical:", f"{theoretical_value:.7f}".rjust(24))

f_circuit = []
num_of_evaluations = 100
for i in range(num_of_evaluations):
    seed = np.random.randint(0, 42424242)

    probs = circuit(seed=seed, return_probs=True)
    samples = circuit(seed=seed)

    bitstring_samples = []
    for sam in samples:
        bitstring_samples.append("".join(str(bs) for bs in sam))

    f_circuit.append(fidelity_xeb(bitstring_samples, probs))
    print(f"\r{i + 1:4d} / {num_of_evaluations:4d}{' ':17}{np.mean(f_circuit):.7f}", end="")
print("\rObserved:", f"{np.mean(f_circuit):.7f}".rjust(27))
'''


