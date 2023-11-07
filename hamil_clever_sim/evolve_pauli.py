from typing import Optional, cast
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli


def evolve_pauli(pauli: Pauli, time: float, label: Optional[str] = None):
    pauli_str = pauli.to_label()
    # ignore any I terms: https://quantumcomputing.stackexchange.com/questions/5567/circuit-construction-for-hamiltonian-simulation?rq=1
    non_id_count = len([c for c in pauli_str if c != "I"])
    qubit_num = pauli.num_qubits
    assert isinstance(qubit_num, int)

    assert pauli.num_qubits is not None

    out: QuantumCircuit
    # 1st case: pauli is just identities (I⊗I⊗I⊗I...)
    if non_id_count == 0:
        out = QuantumCircuit(qubit_num, global_phase=-time)
    elif non_id_count == 1:
        out = get_qiskit_single_rots(pauli, time)
    elif non_id_count == 2:
        out = get_qiskit_dual_rots(pauli, time)
    else:
        out = full_decomp(pauli, time)

    out.name = f"exp(-iHt = {label})"
    return out


def get_qiskit_single_rots(pauli: Pauli, time: float):
    out = QuantumCircuit(cast(int, pauli.num_qubits))

    for i, pauli_c in enumerate(reversed(pauli.to_label())):
        if pauli_c not in ["X", "Y", "Z"]:
            continue

        out.__getattribute__(f"r{pauli_c.lower()}")(2 * time, i)

    return out


def get_qiskit_dual_rots(pauli: Pauli, time: float):
    out = QuantumCircuit(cast(int, pauli.num_qubits))

    labels_arr = np.array(list(reversed(pauli.to_label())))
    qubit_loc = np.where(labels_arr != "I")[0]
    labels = np.array([labels_arr[idx] for idx in qubit_loc])

    joined = f"{labels[0]}{labels[1]}"
    if joined in ["XX", "YY", "ZZ", "ZX", "XZ"]:
        if joined == "XZ":
            out.rzx(2 * time, qubit_loc[1], qubit_loc[0])
        else:
            out.__getattribute__(f"r{joined.lower()}")(
                2 * time, qubit_loc[0], qubit_loc[1]
            )
        return out
    else:
        return full_decomp(pauli, time)


def full_decomp(pauli: Pauli, time: float):
    out = QuantumCircuit(cast(int, pauli.num_qubits))

    cliff = prepare_pauli_clifford(pauli)
    chain = prepare_pauli_chains(pauli)

    # target last non-I pauli
    target = -1
    for i, pauli_c in enumerate(reversed(pauli.to_label())):
        if pauli_c != "I":
            target = i
            break

    assert pauli.num_qubits is not None
    out = QuantumCircuit(pauli.num_qubits)
    out.compose(cliff, inplace=True)
    out.compose(chain, inplace=True)
    out.rz(2 * time, target)
    out.compose(chain.inverse(), inplace=True)
    out.compose(cliff.inverse(), inplace=True)

    return out


def prepare_pauli_clifford(pauli: Pauli):
    assert pauli.num_qubits is not None
    out = QuantumCircuit(pauli.num_qubits)

    for i, pauli_c in enumerate(reversed(pauli.to_label())):
        if pauli_c in ["X", "Y"]:
            if pauli_c == "Y":
                out.sdg(i)
            out.h(i)

    return out


def prepare_pauli_chains(pauli: Pauli):
    assert pauli.num_qubits is not None
    out = QuantumCircuit(pauli.num_qubits)
    con, targ = -1, -1

    for i, pauli_c in enumerate(pauli.to_label()):
        # because of qubits weird endian-ness
        i = pauli.num_qubits - i - 1
        if pauli_c != "I" and con == -1:
            con = i
        elif pauli_c != "I":
            targ = i

        if con >= 0 and targ >= 0:
            out.cx(con, targ)
            con = i
            targ = -1
    return out
