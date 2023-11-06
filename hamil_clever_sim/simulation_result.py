from textual import work
from textual.app import ComposeResult
from textual.containers import (
    HorizontalScroll,
    VerticalScroll,
)
from textual.reactive import Reactive, reactive
from textual.widgets import Label, Markdown, Rule, Static
from textual.worker import Worker
from hamil_clever_sim.components.circuit_drawn import CircuitDrawOutput
from hamil_clever_sim.components.circuit_metadata import CircuitMetadata
from hamil_clever_sim.components.statevector_display import StatevectorDisplay
from hamil_clever_sim.components.timing_data import TimingInformation
from hamil_clever_sim.hamil_runner import (
    SimulationRunner,
    SimulationRunnerResult,
    SimulationTimingData,
)
from string import Template

from hamil_clever_sim.inputs import SimulationKindSet


HEADER_TEMPLATE = Template(
    """
## [$pauli]
with _time =_ $time, _term =_ $n
"""
)


class SimulationHeader(Static):
    classes = "simulation-header"

    def __init__(
        self,
        pauli: str,
        time: str,
        n: str,
        name: str | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id)
        self.pauli = pauli
        self.time = time
        self.n = n

    def compose(self) -> ComposeResult:
        yield Markdown(
            HEADER_TEMPLATE.substitute(time=self.time, pauli=self.pauli, n=self.n)
        )


class SimulationVariantResult(Static):
    job: SimulationRunnerResult
    data: Reactive[dict[str, complex] | None] = reactive(None, layout=True)
    kind: SimulationKindSet
    handle: Worker[None] | None = None
    timing_data: Reactive[SimulationTimingData | None] = reactive(
        None, layout=True, always_update=True
    )

    def create_timing_data_watcher(self):
        def on_timing_update(data: SimulationTimingData):
            self.timing_data = data

        return on_timing_update

    def on_mount(self):
        if self.job is None:
            return

        if self.kind == SimulationKindSet.QC_METHOD:
            info = self.query_one(".simulation-variant-info", HorizontalScroll)
            meta = CircuitMetadata(classes="simulation-variant-info-meta")

            print("[ GETTING METADATA ]")
            meta.data = self.job.get_qc_circuit_metadata(with_draw=True)
            print("[ FINISHED GETTING METADATA ]")
            info.mount(meta)

            draw_zone = CircuitDrawOutput(classes="simulation-variant-info-drawn")
            info.mount(draw_zone)
            draw_zone.data = meta.data.circuit_repr

        self.handle = self.on_job_ready(self.job)
        self.callback = self.create_timing_data_watcher()

    # def watch_job(self, job: SimulationRunnerResult | None) -> None:
    #     if job is None:
    #         return  # waitin on the data...
    #
    #     if self.handle is None:
    #         self.on_job_ready(job)

    @work(thread=True)
    async def on_job_ready(self, job: SimulationRunnerResult):
        data = await job.run(callback=self.callback)

        if self.job is None:
            self.notify(
                f"Job {self.job=} finished, but no data was returned?", severity="error"
            )
            return

        self.data = data

    def watch_data(self, data: dict[str, complex]):
        if data is None:
            return

        statevec = self.query_one(StatevectorDisplay)
        statevec.data = data

    def watch_timing_data(self, data: SimulationTimingData | None):
        if data is None:
            return

        timing = self.query_one(TimingInformation)
        timing.data = data
        timing.watch_data(data)

    def compose(self) -> ComposeResult:
        yield Label(f"via {self.kind}", classes="simulation-variant-header")
        with HorizontalScroll(classes="simulation-variant-info"):
            yield TimingInformation(classes="simulation-variant-info-timing")
        yield StatevectorDisplay()
        yield Rule(line_style="double")


# simulation display will "hold" the
# SimulationRunner object, and then spawns
# children for each kind of simulation to be run.
# We want these to be seperately processed so that
# each result can be streamed in.
class SimulationDisplay(VerticalScroll):
    runner: Reactive[SimulationRunner | None] = reactive(None, layout=True)
    loading = reactive(True)
    classes = "simulation-display"

    async def watch_runner(self, runner: SimulationRunner | None):
        if runner is None:
            return  # nothing to do, yet

        self.loading = False
        header = SimulationHeader(
            "+".join(runner.paulis), str(runner.time), str(runner.n)
        )

        self.mount(header)

        for sim in runner.kind:
            variant_runner = runner.run_job_for_type(sim)
            variant_display = SimulationVariantResult()
            variant_display.kind = sim
            variant_display.job = variant_runner
            self.mount(variant_display)
