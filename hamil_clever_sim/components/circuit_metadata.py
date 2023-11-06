from textual.reactive import Reactive, reactive
from textual.widgets import Static
from rich.table import Table

from hamil_clever_sim.hamil_runner import SimulationCircuitMetadata


class CircuitMetadata(Static):
    data: Reactive[SimulationCircuitMetadata | None] = reactive(None)

    def watch_data(self, update: SimulationCircuitMetadata | None):
        # def on_mount(self) -> None:
        if update is None:
            self.update("<metadata pending...>")
            return

        # log = self.query_one(RichLog)
        grid = Table.grid()
        grid.add_column()
        grid.add_column(justify="right")
        total = 0

        for gate, val in update.gates.items():
            if gate == "barrier":
                continue
            num = f"{val}[i]×{update.factor}[/i]"
            grid.add_row(f"[b]{gate}:[/b]", f"{num}")
            total += val

        grid.add_row("[b]Total gates:[/b]", f"{str(total * update.factor)}")
        self.update(grid)

    # def compose(self) -> ComposeResult:
    #     yield RichLog(
    #         max_lines=8,
    #     )
