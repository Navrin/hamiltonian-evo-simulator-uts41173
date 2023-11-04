from time import strftime

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import Reactive, reactive
from textual.widgets import Label, Static

from hamil_clever_sim.hamil_runner import (
    SimulationTimingData,
)

# TIMING_TEMPLATE = Template(
#     """__Started at:__ ${start}
#
# __Build time:__ ${build_time}s
#
# __Simu. time:__ ${simulation_time}s
#
# __Total time:__ ${total}s
# """
# )


class TimingInformation(Static):
    data: Reactive[SimulationTimingData | None] = reactive(
        None, layout=True, always_update=True
    )
    start = reactive("...")
    build_time = reactive("...")
    simulation_time = reactive("...")
    total_time = reactive("...")

    def watch_data(self, update: SimulationTimingData):
        print(self.data)
        if update is None:
            return

    def compute_build_time(self):
        if self.data is None or self.data.finish_build is None:
            return "..."

        elapsed_ns = self.data.finish_build - self.data.start
        elapsed = elapsed_ns / 1e9
        return str(elapsed) + "s"

    def compute_simulation_time(self):
        if self.data is None:
            return "..."
        deps = [self.data.finish_build, self.data.end_sim]

        missing = any([dep is None for dep in deps])
        if missing:
            return "..."

        assert self.data.finish_build is not None
        assert self.data.end_sim is not None

        elapsed_ns = self.data.end_sim - self.data.finish_build
        elapsed = elapsed_ns / 1e9
        return str(elapsed) + "s"

    def compute_total_time(self):
        if self.data is None or self.data.end_sim is None:
            return "..."

        elapsed_ns = self.data.end_sim - self.data.start
        elapsed = elapsed_ns / 1e9
        return str(elapsed) + "s"

    def compute_start(self):
        if self.data is None:
            return "..."

        return strftime("%H:%M:%S - %d/%m", self.data.start_as_timestamp)

    # def compute_template(self) -> str:
    #     return TIMING_TEMPLATE.substitute(
    #         start=self.start,
    #         build_time=self.build_time,
    #         simulation_time=self.simulation_time,
    #         total=self.total_time,
    #     )

    def watch_start(self, value: str):
        self.query_one(".timing-data-start", Label).update(Text(value))

    def watch_build_time(self, value: str):
        self.query_one(".timing-data-build", Label).update(Text(value))

    def watch_simulation_time(self, value: str):
        self.query_one(".timing-data-simulation", Label).update(Text(value))

    def watch_total_time(self, value: str):
        self.query_one(".timing-data-total", Label).update(Text(value))

    def compose(self) -> ComposeResult:
        with Horizontal(classes="timing-data-elem"):
            yield Label("Started at:", classes="timing-data-label")
            yield Label("...", classes="timing-data-start")
        with Horizontal(classes="timing-data-elem"):
            yield Label("Build time:", classes="timing-data-label")
            yield Label("...", classes="timing-data-build")
        with Horizontal(classes="timing-data-elem"):
            yield Label("Simu. time:", classes="timing-data-label")
            yield Label("...", classes="timing-data-simulation")
        with Horizontal(classes="timing-data-elem"):
            yield Label("Total time:", classes="timing-data-label")
            yield Label("...", classes="timing-data-total")
