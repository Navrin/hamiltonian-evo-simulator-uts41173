from itertools import product

import pytest
import qiskit as q
import qiskit.quantum_info as qi
from qiskit.primitives import Estimator
from qiskit.quantum_info.operators import Pauli
from qiskit.synthesis.evolution.product_formula import (
    evolve_pauli as qiskit_evolve_pauli,
)
from qiskit_aer import Aer, AerSimulator, StatevectorSimulator
from qiskit_algorithms.time_evolvers import TimeEvolutionProblem
from qiskit_algorithms.time_evolvers.trotterization import TrotterQRTE

from hamil_clever_sim.evolve_pauli import evolve_pauli

paulis = ["I", "X", "Y", "Z"]


# def batch(iter, n):
#     return [("".join(list(it[:n])), "".join(list(it[n:]))) for it in iter]
#

# create_pauli_str = lambda n: list(
#     batch(combinations_with_replacement(paulis, n), n // 2)
# )
#
create_pauli_str = lambda n: list(["".join(x) for x in product(paulis, repeat=n)])  # noqa: E731


pauls_1 = create_pauli_str(1)
pauls_2 = create_pauli_str(2)
pauls_3 = create_pauli_str(3)
t = [1, 2, 3, 4, 5, 6]


def obtain_qiskit_implementation_result(p1, p2, t, N=4):
    assert len(p1) == len(p2)
    estimator = Estimator()
    init = q.QuantumCircuit(len(p1))

    op = qi.SparsePauliOp([qi.Pauli(p1), qi.Pauli(p2)])
    evolutor = TimeEvolutionProblem(op, t, init)
    trotter_qrte = TrotterQRTE(estimator=estimator, num_timesteps=N)
    evolved_state = trotter_qrte.evolve(evolutor).evolved_state

    # returns statevector, therefore we need to simulate with the statevector backend
    return evolved_state


PRECISION_FOR_TESTS = 3


class TestPauliEvoImplementation:
    # reuse the same simulator to minimise test time
    @pytest.fixture(scope="session")
    def backend(self) -> AerSimulator:
        # backend = Aer.get_backend("statevector_simulator")
        backend = AerSimulator(method="statevector")
        return backend

    def check_our_pauli_implementation_matches(self, t: float, pauli: Pauli, backend):
        our_impl = evolve_pauli(pauli, t)
        qiskit_impl = qiskit_evolve_pauli(pauli, t)
        our_impl.save_statevector()  # type:ignore
        qiskit_impl.save_statevector()  # type:ignore

        job_our = backend.run(our_impl)
        job_qis = backend.run(qiskit_impl)

        res_our = job_our.result()
        res_qis = job_qis.result()

        our_statevec = res_our.get_statevector(decimals=PRECISION_FOR_TESTS)
        qis_statevec = res_qis.get_statevector(decimals=PRECISION_FOR_TESTS)
        print(f"{our_statevec=}")
        print(f"{qis_statevec=}")

        assert our_statevec == qis_statevec

    @pytest.mark.parametrize("p", pauls_1)
    @pytest.mark.parametrize("t", t)
    def test_one_level_paulis_match_qiskit_evo(
        self, p, t, backend: StatevectorSimulator
    ):
        self.check_our_pauli_implementation_matches(t, Pauli(p), backend)

    @pytest.mark.parametrize("p", pauls_2)
    @pytest.mark.parametrize("t", t)
    def test_two_level_paulis_match_qiskit_evo(self, p, t, backend):
        self.check_our_pauli_implementation_matches(t, Pauli(p), backend)

    @pytest.mark.parametrize("p", pauls_3)
    @pytest.mark.parametrize("t", t)
    def test_three_level_paulis_match_qiskit_evo(self, p, t, backend):
        self.check_our_pauli_implementation_matches(t, Pauli(p), backend)
