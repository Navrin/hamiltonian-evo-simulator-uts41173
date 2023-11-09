from itertools import product
from typing import Tuple

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.reactive import Reactive, reactive
from textual.widgets import (
    DataTable,
    Static,
)
from textual.widgets.data_table import ColumnKey

from hamil_clever_sim.components.sparkline_controllable import Sparkline
from hamil_clever_sim.hamil_runner import SimulationRunnerResult

COLS = ("|𝜓⟩", "amplitude (ℝ)", "amplitude (ℂ)", "probability (%)")


class StatevectorDisplay(Static):
    data: Reactive[SimulationRunnerResult.Data |
                   None] = reactive(None, layout=True)
    rows: Reactive[list[tuple[str, str, str]]] = reactive([])
    column_keys: list[ColumnKey] = []
    spark_data: Reactive[list[float]] = reactive([])
    all_states = reactive([])

    marked_index = reactive(-1)

    def on_mount(self):
        table = self.query_one(DataTable)
        self.column_keys = table.add_columns(*COLS)

    def compute_rows(self):
        if self.data is None:
            return []
        # print(list(self.data.items()))
        rows = [
            (
                f"|{label}⟩",
                Text("{: .5f}".format(entry[0].real), justify="right"),
                "{:+.5f}𝕚".format(entry[0].imag),
                Text("{:.2f}%".format(entry[1] * 100), justify="right"),
            )
            for label, entry in self.data.items()
        ]
        return rows

    def watch_data(self, update: dict[str, Tuple[complex, float]]):
        if update is None:
            return

    def watch_rows(self, update: list[tuple[str, str, str]]) -> None:
        table = self.query_one(DataTable)
        for row in update:
            table.add_row(*row)
        spark = self.query_one(Sparkline)
        spark.data = self.spark_data

    def compute_all_states(self):
        if self.data is None:
            return []
        el = next(iter(self.data.keys()))
        states = ["".join(state) for state in product("01", repeat=len(el))]
        return states

    def compute_spark_data(self):
        states = self.all_states
        if len(states) == 0:
            return

        assert self.data is not None

        return [abs((self.data.get(state) or (0,))[0]) for state in states]

    @on(DataTable.RowHighlighted)
    def handle_row_select(self, highlight: DataTable.RowHighlighted):
        cursor = highlight.cursor_row
        states = self.all_states
        as_state = highlight.data_table.get_row_at(
            cursor)[0][1:-1]  # to remove the ket
        marked_cursor = states.index(as_state)

        self.marked_index = marked_cursor
        spark = self.query_one(Sparkline)
        spark.marked_index = marked_cursor
        spark.refresh()
        print(spark.marked_index)

    def compose(self) -> ComposeResult:
        yield Sparkline(self.spark_data)
        yield DataTable(cursor_type="row", classes="statevector-display-table")
