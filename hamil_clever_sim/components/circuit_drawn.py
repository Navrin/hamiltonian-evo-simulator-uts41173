from typing import Optional
from qiskit.visualization.circuit.text import TextDrawing
from textual.containers import VerticalScroll
from textual.reactive import Reactive, reactive
from textual.widgets import Label


class CircuitDrawOutput(VerticalScroll):
    data: Reactive[Optional[TextDrawing]] = reactive(None)

    def watch_data(self, update: Optional[TextDrawing]):
        if update is None:
            return

        self.mount(Label(update.single_string()))
