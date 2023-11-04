from time import gmtime, strftime
from typing import Any
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.reactive import Reactive, reactive
from textual.widget import Widget
from textual.widgets import Label, LoadingIndicator, Markdown, Pretty, Rule, Static
from textual.worker import Worker
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
        yield TimingInformation()
        # Reimplement the cool sparkliner :(
        yield StatevectorDisplay()
        yield Rule(line_style="double")


# simulation display will "hold" the
# SimulationRunner object, and then spawns
# children for each kind of simulation to be run.
# We want these to be seperately processed so that
# each result can be streamed in.
class SimulationDisplay(ScrollableContainer):
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
