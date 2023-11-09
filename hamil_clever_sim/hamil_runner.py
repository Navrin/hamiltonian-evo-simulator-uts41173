"""
This module provides the core simulation and data handling processes for the app.
We expose a few helper classes to act as the "Model" of the data that the view layer can use.

"""

from __future__ import annotations

import asyncio
import time
import typing as ty
from asyncio import Task

import numpy as np
import qiskit as q
from qiskit import quantum_info as qi
from qiskit.circuit import Instruction
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit.visualization.circuit.circuit_visualization import _text_circuit_drawer
from qiskit.visualization.circuit.text import TextDrawing
from qiskit_aer import AerSimulator
from scipy.linalg import expm
from typing_extensions import Callable, Optional, OrderedDict, Tuple, cast

from hamil_clever_sim.evolve_pauli import evolve_pauli
from hamil_clever_sim.inputs import PauliStringValidator, SimulationKindSet


def create_u(op: str, coef: Optional[float] = None):
    """Implements operator U=exp(iHt) for a given pauli term, with an optional coefficient.

    :param op: A valid pauli string, single term only.
    :param coef: A coefficient in the form of a float, that will multiply the time value (from relation $e^{ic_kP_kt}$)
    :return: A closure that of the form U(t), generating a circuit for a given time value.
    :rtype: (time: float, label: Optional[str]) -> QuantumCircuit
    """
    weight = coef if coef is not None else 1.0

    def u_n(t, label=None):
        weighted_t = t * weight
        paul = evolve_pauli(qi.Pauli(op), weighted_t, label=label)
        if label is None:
            label = f"$U({t} * {weight})$"
        # paul.name = f"{label}={paul.name}"
        paul.name = "unitary_evolve"
        return paul

    return u_n


def build_iterative_circuit(t: float, *u_states, N: int = 4):
    r"""Implements a trotterised circuit from a vararg of U(t) circuits.
    This method assumes that given trotterised circuits of X+Y and X+Y+Z are valid, then
    X+Y+Z+X+...+Z are also valid.


    :param N: Trotterisation terms / simulation accuracy. Higher values are better, but will balloon the gate count.
    :param t: How many time steps to simulate for. This term does not introduce more gates, just what the rotation value will be.
    :return: Returns a circuit that implements $\ket{\psi_t}$
    :rtype: :class:`QuantumCircuit`
    """
    # start = time.perf_counter_ns()
    u_1 = u_states[0]
    print(u_1(t).num_qubits)
    circ = q.QuantumCircuit(u_1(t).num_qubits)
    circ.barrier(label="n=0")
    # NOTE: don't use the for_loop interface, it seems
    # to only be of use for when we are performing actions based on
    # some sort of conditional around the loop within the circuit.
    # otherwise it spawns a bunch of parameters and causes issues when
    # trying to decompose the circuit.
    for n in range(1, N + 1):
        states = list(zip(u_states, range(1, len(u_states) + 1)))
        for u, i in states:
            circ.append(u(t / N, label=f"U_{i}({t} / {N})"), range(0, circ.num_qubits))
        circ.barrier(label=f"n={n}")

    # end = time.perf_counter_ns()
    # print(f"Construction of iterated circuit took {end - start} ns")
    circ.save_statevector()  # type:ignore

    return circ


def build_operator_circuit(
    t, *u_states, N: int = 4, qubits: Optional[int] = None
) -> Operator:
    r"""This method implements trotterisation a hamiltonian while also not needing to create a copy of the
    circuit N times. Instead, the first step of the simulation is taken, and then converted into a unitary operator
    of form $UU^\dagger=I$. This operator is then taken to the power of N, turning the problem into an optimised linalg computation.
    This operator can then be fed directly into a prepared :class:`Statevector`, using the :func:`Statevector.evolve` method.

    :param N: Trotterisation terms / simulation accuracy. Higher values are better, but will balloon the gate count.
    :param t: How many time steps to simulate for. This term does not introduce more gates, just what the rotation value will be.
    :param qubits: The largest qubit value, in case not all pauli terms are of equal dimensions.
    :return: An operator that can implement $\ket{\phi_t}$ for a statevector.
    """
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


def statevec_backend() -> AerSimulator:
    # apparently the statevector sim is being depreciated, thanks qiskit!
    # backend = statevector_simulator.StatevectorSimulator()
    backend = AerSimulator(method="statevector")
    return backend


backend = statevec_backend()


class SimulationCircuitMetadata:
    """Data model for optional quantum circuit metadata.
    For displaying the overall gate complexity, and also the
    drawn form of the circuit.

    :param factor: Represents terms N
    :param gates: Dictionary of gate -> count mappings
    :param circuit_repr: Holds the qiskit circuit drawer
    """

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
    """Data model for live updating timing information.
    This model can be updated in real time, as the simulator/builder progresses
    through the steps of simulation. This model also acts as a loading bar, to make sure
    that the user knows the program has not crashed/stalled.

    :param start: When the simulation job began.
    :param start_as_timestamp: For displaying the actual timestamp, as perf_counter_ns is tricky to turn into a timestamp.
    :param update_callback: For letting the view model react to internal changes within the class.
    :param finish_build: When the build was completed, and when the simulation started.
    :param end_sim: When the simulation ended, and results processed.
    """

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

    def sim_ended(self):
        self.end_sim = time.perf_counter_ns()
        self.register_update()


class SimulationRunnerResult:
    """
    This class encapsulates the process of running specific kind of simulation,
    and processing the output into a standard form that can be displayed via the
    :class:`~hamil_clever_sim.components.statevector_display.StatevectorDisplay` widget.

    Generating a class per-type allows us to handle each siulation async/on worker threads,
    such that we can get the output of the much faster methods to display, while the slower methods
    work in the background.

    :param meta: The :class:`~hamil_clever_sim.hamil_runner.SimulationRunner` that spawned this runner.
    :param timing_data: A :class:`~hamil_clever_sim.hamil_runner.SimulationTimingData` that is updated for each stage of the simulation.
    :param type: A single member bitflag from :class:`~hamil_clever_sim.inputs.SimulationKindSet` that represents the type of simulation this runner will use.
    :param data: The final statevector output given after the simulation completes.
                 The data will be in the form of a key value pair, with the key being a ket notation of a state (i.e 001, 010, 110)
                 and the value will be a two tuple, with the first element being the complex amplitude, and the second as the probability out of 100% that this state will be measured.
    """

    Data = dict[str, Tuple[complex, float]]
    type: SimulationKindSet
    meta: SimulationRunner
    data: Data
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

    def process(self, data: Statevector) -> Data:
        out = data.to_dict(decimals=self.meta.OUTPUT_PRECISION)
        probabilities = data.probabilities_dict(decimals=self.meta.OUTPUT_PRECISION)

        for key, prob in probabilities.items():
            out[key] = (out[key], prob)

        self.data = out
        return self.data

    async def run(self, callback=None):
        """Main entrypoint for the simulation runner, will automatically
        use the correct simulation for the given kind of runner.
        This method was moved from the parent class, as we wanted access to the timing data,
        while also having each runner seperate form each other. This is so we can run the simulations
        on different threads.

        :param callback: Timing data callbacker, for the view layer
        """
        self.timing_data.register_callback(callback)
        if self.type == SimulationKindSet.QC_METHOD:
            self.get_qc_circuit_metadata()
            return await self.run_qc_simulation()
        elif self.type == SimulationKindSet.OP_METHOD:
            return await self.run_op_simulation()
        else:
            return await self.run_direct_calc()

    async def run_qc_simulation(self):
        """Run the quantum siulation in an async fashion, polling the backend every 0.15s."""
        runner = self.meta

        paulis_circ = [
            create_u(term, float(coef) if coef is not None else 1)
            for term, coef in runner.weighted_paulis
        ]

        circ = build_iterative_circuit(runner.time, *paulis_circ, N=runner.n)
        self.timing_data.build_finished()

        job = backend.run(circ.decompose(reps=2))

        async def poll_job(interval=0.15):
            try:
                while job.running():
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
        r"""Generate the operator and evolve a zero-initialised statevector $\ket{0}^{\otimes n}$"""
        runner = self.meta

        paulis_circ = [
            create_u(term, float(coef) if coef is not None else 1)
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
            create_u(term, float(coef) if coef is not None else 1)
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
        # yes, even calling these methods directly
        # and setting the encoding doesn't work.
        # it seems to really want the stdout handle,
        # but we are in a tui context and need to fake it.
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

    u1 = create_u(p1, 3)
    u2 = create_u(p2, 5)

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

    print("[statevector probabilities]", pprint.pformat(result.probabilities_dict()))
