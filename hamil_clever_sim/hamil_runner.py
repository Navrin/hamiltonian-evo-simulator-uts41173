from __future__ import annotations
from asyncio import Task
import asyncio
import time
import typing as ty
from typing_extensions import Callable, Optional, OrderedDict, cast

import numpy as np
import qiskit as q
from qiskit import quantum_info as qi
from qiskit.circuit import Instruction
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

# this gate implements the U(t) = exp(-itH) operator
# from qiskit.extensions import HamiltonianGate
# from qiskit.circuit.library import PauliEvolutionGate as PauliEvGate
from qiskit.synthesis.evolution.product_formula import evolve_pauli
from qiskit.visualization.circuit.text import TextDrawing
from qiskit_aer.backends import StatevectorSimulator
from scipy.linalg import expm

from hamil_clever_sim.inputs import PauliStringValidator, SimulationKindSet
from qiskit.visualization.circuit.circuit_visualization import _text_circuit_drawer


# TODO: reimplement this code and test it against the qiskit implementation to
# finally "extract" the evolve_pauli method
# from https://github.com/DavitKhach/quantum-algorithms-tutorials/blob/master/Hamiltonian_simulation.ipynb
# and https://medium.com/quantum-untangled/hamiltonian-simulation-with-quantum-computation-fedc7cdc02e0
# !! this should not be left as-is for final submission !!
# !! this code is here for reference in case any of these links go dark !!
def sim_z(t, qc, qubits):
    for i in range(len(qubits) - 1):
        qc.cx(qubits[i], qubits[i + 1])
    qc.rz(-2 * t, qubits[-1])
    for i in range(len(qubits) - 1, 0, -1):
        qc.cx(qubits[i - 1], qubits[i])


def sim_pauli(arr, t, qc, qubits, coef=1):
    new_arr = []
    new_qub = []
    for idx in range(len(arr)):
        if arr[idx] != "I":
            new_arr.append(arr[idx])
            new_qub.append(qubits[idx])

    h_y = 1 / np.sqrt(2) * np.array([[1, -1j], [1j, -1]])
    for i in range(len(new_arr)):
        if new_arr[i] == "X":
            qc.h(new_qub[i])
        elif new_arr[i] == "Y":
            qc.unitary(h_y, [new_qub[i]], r"$H_y$")

    sim_z(t, qc, new_qub)

    for i in range(len(new_arr)):
        if new_arr[i] == "X":
            qc.h(new_qub[i])
        elif new_arr[i] == "Y":
            qc.unitary(h_y, [new_qub[i]], r"$H_y$")


def build_operator_circuit(t, *u_states, N=4, qubits=None) -> Operator:
    start = time.perf_counter_ns()
    # we assume all u_gates are of equal dims if qubits arg is not passed
    u_1 = u_states[0]

    circ = q.QuantumCircuit(qubits if qubits is not None else u_1(t).num_qubits)

    for u, i in zip(u_states, range(1, len(u_states) + 1)):
        circ.append(u(t / N, label=f"U_{i}({t} / {N})"), range(0, circ.num_qubits))

    np_op = Operator(circ).power(N)

    end = time.perf_counter_ns()

    print(f"Construction of operator circuit took {end-start} ns")
    return np_op


def create_u(op):
    def u_n(t, label=None):
        paul = evolve_pauli(qi.Pauli(op), t, label=label)
        if label is None:
            label = f"$U_1({t})$"
        # paul.name = f"{label}={paul.name}"
        paul.name = "unitary_evolve"
        return paul

    return u_n


def create_u_with_coef(op, coef: float):
    weight = coef

    def u_n(t, label=None):
        weighted_t = t * weight
        paul = evolve_pauli(qi.Pauli(op), weighted_t, label=label)
        if label is None:
            label = f"$U_1({t} * {weight})$"
        # paul.name = f"{label}={paul.name}"
        paul.name = "unitary_evolve"

        return paul

    return u_n


def build_iterative_circuit(t, *u_states, N=4):
    start = time.perf_counter_ns()
    u_1 = u_states[0]
    print(u_1(t).num_qubits)
    circ = q.QuantumCircuit(u_1(t).num_qubits)
    circ.barrier(label="n=0")
    for n in range(1, N + 1):
        states = list(zip(u_states, range(1, len(u_states) + 1)))
        print(f"{list(states)=} {repr(u_states)=}")
        for u, i in states:
            print(f"{u=} {i=}")
            circ.append(u(t / N, label=f"U_{i}({t} / {N})"), range(0, circ.num_qubits))
        # circ.append(u_1(t/N, label=f"$U_1({t} / {N} )$"), range(0, circ.num_qubits))
        # if u_2 is not None:
        #     circ.append(u_2(t/N, label=f"$U_2({t} / {N} )$"), range(0, circ.num_qubits))
        # if u_3 is not None:
        #     circ.append(u_3(t/N, label=f"$U_3({t} / {N} )$"), range(0, circ.num_qubits))
        circ.barrier(label=f"n={n}")

    end = time.perf_counter_ns()
    print(f"Construction of iterated circuit took {end - start} ns")

    return circ


def statevec_backend() -> StatevectorSimulator:
    from qiskit_aer import statevector_simulator

    backend = statevector_simulator.StatevectorSimulator()
    return backend


backend = statevec_backend()


class SimulationCircuitMetadata:
    factor: int
    gates: ty.OrderedDict[str, int]
    # circuit_repr: ty.Optional[Callable[[], TextDrawing]]
    circuit_repr: Optional[TextDrawing]

    def __init__(
        self,
        gates: ty.OrderedDict[Instruction, int],
        factor: int,
        drawn: Optional[Callable[[], TextDrawing]] = None,
    ) -> None:
        self.factor = factor
        self.gates = OrderedDict(
            [(str(instruction), val) for instruction, val in gates.items()]
        )
        self.circuit_repr = drawn() if drawn is not None else None

    def get_counts(self) -> ty.Iterable[str]:
        return iter(
            [f"{gate}: {val}×{self.factor}" for gate, val in self.gates.items()]
        )


class SimulationTimingData:
    start_as_timestamp = time.localtime()

    start: int = time.perf_counter_ns()
    finish_build: int | None = None
    # start_sim:int | None = None
    end_sim: int | None = None
    update_callback: ty.Callable[[ty.Any], None] | None

    def __init__(self, callback=None) -> None:
        self.start = time.perf_counter_ns()
        self.start_as_timestamp = time.localtime()
        if callback is not None:
            self.update_callback = callback

    def register_callback(self, callback):
        self.update_callback = callback

    def register_update(self):
        if self.update_callback is not None:
            self.update_callback(self)

    def build_finished(self):
        self.finish_build = time.perf_counter_ns()
        self.register_update()

    def __repr__(self):
        return (
            f"SimulationTimingData(\nstart={self.start},"
            + f"\nfinish_build={self.finish_build},\n"
            + f"\nend_sim={self.end_sim})"
        )

        # we can assume running the simulation is the next
        # step after building the circuit

    # def sim_started(self):
    #     self.start_sim = time.perf_counter_ns()

    def sim_ended(self):
        self.end_sim = time.perf_counter_ns()
        self.register_update()


class SimulationRunnerResult:
    type: SimulationKindSet
    meta: SimulationRunner
    data: dict[str, complex]
    job: Task[Statevector]

    def __init__(self, meta: SimulationRunner):
        self.meta = meta
        # starting the timer at initiation.
        # technically bad behaviour because
        # the sim/building may not have started just yet
        # but it makes it a lot easier to directly bind
        # data to the view layer.
        self.timing_data = SimulationTimingData()

    def set_type(self, type: SimulationKindSet):
        self.type = type

    def process(self, data: Statevector) -> dict[str, complex]:
        self.data = data.to_dict(decimals=self.meta.OUTPUT_PRECISION)
        return self.data

    async def run(self, callback=None):
        self.timing_data.register_callback(callback)
        if self.type == SimulationKindSet.QC_METHOD:
            self.get_qc_circuit_metadata()
            return await self.run_qc_simulation()
        elif self.type == SimulationKindSet.OP_METHOD:
            return await self.run_op_simulation()
        else:
            return await self.run_direct_calc()

    async def run_qc_simulation(self):
        runner = self.meta

        paulis_circ = [
            create_u_with_coef(term, float(coef) if coef is not None else 1)
            for term, coef in runner.weighted_paulis
        ]

        circ = build_iterative_circuit(runner.time, *paulis_circ, N=runner.n)
        self.timing_data.build_finished()

        job = backend.run(circ.decompose(reps=2))

        async def poll_job(interval=0.15):
            try:
                while job.running():
                    print("Polling = " + str(job.running))
                    await asyncio.sleep(interval)
                return job.result()
            finally:
                if not job.in_final_state():
                    job.cancel()

        result = await poll_job()
        self.timing_data.sim_ended()

        st = result.get_statevector()
        assert isinstance(st, Statevector)
        return self.process(st)

    async def run_op_simulation(self):
        runner = self.meta

        paulis_circ = [
            create_u_with_coef(term, float(coef) if coef is not None else 1)
            for term, coef in runner.weighted_paulis
        ]
        largest = max([pc(runner.time).num_qubits for pc in paulis_circ])
        op = build_operator_circuit(
            runner.time, *paulis_circ, N=runner.n, qubits=largest
        )
        self.timing_data.build_finished()

        init_state = Statevector.from_label("0" * largest)
        res = init_state.evolve(op)
        self.timing_data.sim_ended()

        return self.process(res)

    async def run_direct_calc(self):
        runner = self.meta

        paulis_circ = [
            (term, float(coef) if coef is not None else 1)
            for term, coef in runner.weighted_paulis
        ]
        largest = max([len(term) for term, _ in paulis_circ])
        init_state = Statevector.from_label("0" * largest)

        pauli_collect = cast(tuple[tuple[str], tuple[float]], tuple(zip(*paulis_circ)))
        pauli_sparse = SparsePauliOp(list(pauli_collect[0]), np.array(pauli_collect[1]))
        self.timing_data.build_finished()

        exp = expm(-1j * pauli_sparse.to_matrix() * self.meta.time)
        self.timing_data.sim_ended()
        out = Statevector(np.matmul(exp, init_state))

        return self.process(out)

    def get_qc_circuit_metadata(self, with_draw=False):
        runner = self.meta

        paulis_circ = [
            create_u_with_coef(term, float(coef) if coef is not None else 1)
            for term, coef in runner.weighted_paulis
        ]

        circ = build_iterative_circuit(runner.time, *paulis_circ, N=1)

        decomped = circ.decompose(
            ["rzx", "rxz", "rzz", "ryy", "rxx", "unitary_evolve"], reps=2
        )
        counted = decomped.count_ops()

        # qiskit is mega dumb, as per usual
        # defer attempting to "draw" until we are
        # able to put it in a fake stdout context
        def defer_drawing():
            drawing = _text_circuit_drawer(
                decomped,
                initial_state=False,
                vertical_compression="high",
                plot_barriers=False,
                fold=-1,
                with_layout=False,
                encoding="utf8",
            )
            return drawing

        return SimulationCircuitMetadata(
            counted,
            self.meta.n,
            drawn=defer_drawing,
        )


class SimulationRunner:
    OUTPUT_PRECISION = 4

    def __init__(
        self, pauli: str, time: float, n: int, kind: SimulationKindSet
    ) -> None:
        self.paulis = pauli.split("+")
        self.weighted_paulis = []
        for split_pauli in self.paulis:
            match = PauliStringValidator.pauli_string_regex.fullmatch(split_pauli)
            assert match is not None
            coef = match.group("coef")
            term = match.group("term")

            self.weighted_paulis.append((term, coef))

        print(repr(self.weighted_paulis))

        self.time = time
        self.n = n
        self.kind = kind

    def run_job_for_type(self, kind: SimulationKindSet) -> SimulationRunnerResult:
        assert len(kind) == 1
        resulter = SimulationRunnerResult(self)

        if bool(kind & SimulationKindSet.QC_METHOD):
            resulter.set_type(SimulationKindSet.QC_METHOD)
        elif bool(kind & SimulationKindSet.OP_METHOD):
            resulter.set_type(SimulationKindSet.OP_METHOD)
        elif bool(kind & SimulationKindSet.EXP_METHOD):
            resulter.set_type(SimulationKindSet.EXP_METHOD)
        else:
            raise ValueError(f"{kind} does not contain any correct values")

        return resulter


if __name__ == "__main__":
    p1 = "XZ"
    p2 = "YX"
    N = 10
    t = 10

    init_state = Statevector.from_label("0" * len(p1))

    u1 = create_u_with_coef(p1, 3)
    u2 = create_u_with_coef(p2, 5)

    start_1 = time.perf_counter_ns()
    sim = build_iterative_circuit(t, u1, u2, N=N)
    job = backend.run(sim.decompose(reps=2))
    result = job.result().get_statevector()
    end_1 = time.perf_counter_ns()

    start_2 = time.perf_counter_ns()
    op = build_operator_circuit(t, u1, u2, N=N)
    res = init_state.evolve(op)
    end_2 = time.perf_counter_ns()
    print(f"Simulating via iterative method took {(end_1 - start_1) / 1000} ms")
    print(f"Simulating via operator method took  {(end_2 - start_2) / 1000} ms")
    import pprint

    assert isinstance(result, Statevector)
    print("[iterated method] \n", pprint.pformat(result.to_dict()))
    print(
        f"[operator method (time ratio {(end_1 - start_1)/(end_2-start_2) }% faster)] = \n",
        pprint.pformat(res.to_dict()),
    )
