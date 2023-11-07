from itertools import combinations_with_replacement

import pytest
import qiskit as q
from qiskit import QuantumCircuit
from qiskit import quantum_info as qi
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer, AerSimulator
from hamil_clever_sim.hamil_runner import build_iterative_circuit, create_u
from qiskit.primitives import Estimator
from qiskit_algorithms.time_evolvers import TimeEvolutionProblem
from qiskit_algorithms.time_evolvers.trotterization import TrotterQRTE

paulis = ["I", "X", "Y", "Z"]

# this essentially defines the error bounds,
# trauncates the decimals of the statevector
precision = 7


def batch(iter, n):
    return [("".join(list(it[:n])), "".join(list(it[n:]))) for it in iter]


create_pauli_str = lambda n: list(  # noqa: E731
    batch(combinations_with_replacement(paulis, n), n // 2)
)


def obtain_qiskit_implementation_result(p1, p2, t, N=4):
    assert len(p1) == len(p2)
    estimator = Estimator()
    init = QuantumCircuit(len(p1))

    op = qi.SparsePauliOp([qi.Pauli(p1), qi.Pauli(p2)])
    evolutor = TimeEvolutionProblem(op, t, init)
    trotter_qrte = TrotterQRTE(estimator=estimator, num_timesteps=N)
    evolved_state = trotter_qrte.evolve(evolutor).evolved_state

    return Statevector(evolved_state)


# reuse the same simulator to minimise test time


pauls_1 = create_pauli_str(2)
pauls_2 = create_pauli_str(4)
pauls_3 = create_pauli_str(6)
t = [1, 2, 3, 4, 5, 6]


class TestPauliTrotterImplementation:
    @pytest.fixture(scope="session")
    def statevec_backend(self):
        # backend = Aer.get_backend("statevector_simulator")
        backend = AerSimulator(method="statevector")
        return backend

    def trotter_is_same_over_n_pauli_set(self, t, paulis, backend, N=4):
        p1, p2 = paulis

        u1 = create_u(p1, coef=1.0)
        u2 = create_u(p2, coef=1.0)
        print(type(t))

        ours = build_iterative_circuit(t, u1, u2, N=N)
        verif_state = obtain_qiskit_implementation_result(p1, p2, t, N=N)
        # assert isinstance(verif, QuantumCircuit)
        assert isinstance(verif_state, Statevector)

        our_job = backend.run(ours.decompose(reps=1))
        our_results = our_job.result()
        # verif_job = backend.run(verif.decompose(reps=2))
        # verif_results = verif_job.result()

        our_state = our_results.get_statevector(decimals=precision)
        # verif_state = verif_results.get_statevector(decimals=precision)
        print(
            f'PAULIS ({p1}, {p2}) -> {our_state.draw(output="text")} ==? {verif_state.draw(output="text")}'
        )
        assert our_state == verif_state

    @pytest.mark.parametrize("t", t)
    @pytest.mark.parametrize("p", pauls_1)
    def test_trotter_is_same_over_single_pauli_set(self, t, p, statevec_backend):
        self.trotter_is_same_over_n_pauli_set(t, p, statevec_backend)

    # @pytest.mark.skip()
    @pytest.mark.parametrize("t", t)
    @pytest.mark.parametrize("p", pauls_2)
    def test_trotter_is_same_over_two_pauli_set(self, t, p, statevec_backend):
        self.trotter_is_same_over_n_pauli_set(t, p, statevec_backend)

    # @pytest.mark.skip()
    @pytest.mark.parametrize("t", t)
    @pytest.mark.parametrize("p", pauls_3)
    def test_trotter_is_same_over_three_pauli_set(self, t, p, statevec_backend):
        self.trotter_is_same_over_n_pauli_set(t, p, statevec_backend)

    @pytest.mark.parametrize("t", t)
    @pytest.mark.parametrize("p", pauls_1)
    @pytest.mark.parametrize("n", range(3, 8))
    def test_single_paulis_of_various_t_and_n(self, t, p, n, statevec_backend):
        self.trotter_is_same_over_n_pauli_set(t, p, statevec_backend, n)
