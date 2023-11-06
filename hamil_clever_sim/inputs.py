from enum import IntFlag, auto
from typing import cast
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Blur, DescendantBlur
from textual.message import Message
from textual.validation import Integer, Number, Regex
from textual.widgets import Button, Input, Label, SelectionList, Static
from textual.widgets.selection_list import Selection
from textual.reactive import Reactive, reactive
import re


class PauliStringValidator:
    pauli_string_regex = re.compile(r"^(?!\+)((\+)*((?P<coef>\d)*(?P<term>[XYZ]+)))+$")


class SimulationKindSet(IntFlag):
    QC_METHOD = auto()
    OP_METHOD = auto()
    EXP_METHOD = auto()

    def description(self) -> str:
        if self == self.QC_METHOD:
            return "Quantum Circuit Composition"
        if self == self.OP_METHOD:
            return "Fast Operator Method"
        if self == self.EXP_METHOD:
            return "Direct Numeric Calculation"
        else:
            return "Not a valid variant!"

    def to_selection(self) -> Selection:
        return Selection(f"via {self.description()}", self)

    def __str__(self) -> str:
        return self.description()


class InputForm(Static):
    pauli_input: Reactive[str | None] = reactive(None)
    time_input: Reactive[float | None] = reactive(None)
    n_input: Reactive[int | None] = reactive(None)
    kind_selection_input: Reactive[SimulationKindSet] = reactive(SimulationKindSet(0))

    SELECTIONS = map(SimulationKindSet.to_selection, iter(SimulationKindSet))

    def compose(self) -> ComposeResult:
        with Vertical(id="input-form"):
            with Vertical(id="input-form-pauli", classes="input-form-input"):
                yield Label(
                    "Hamiltonian: ",
                    id="input-form-pauli-label",
                    classes="input-form-label",
                )
                yield Input(
                    id="input-form-pauli-input",
                    classes="input-form-input-component",
                    placeholder="As Pauli (i.e X+Y or 3YZ+4YX)",
                    validators=[
                        Regex(
                            PauliStringValidator.pauli_string_regex,
                            failure_description="Not a valid Pauli (case-sensitive.)",
                        ),
                    ],
                    validate_on=["blur", "submitted"],
                )
                yield Label("", id="input-form-pauli-error", classes="input-form-error")

            with Vertical(id="input-form-time", classes="input-form-input"):
                yield Label(
                    "Time steps: ",
                    id="input-form-time-label",
                    classes="input-form-label",
                )
                yield Input(
                    id="input-form-time-input",
                    placeholder="Enter time to simulate (float)",
                    classes="input-form-input-component",
                    validators=[Number(minimum=0)],
                    validate_on=["blur", "submitted"],
                )
                yield Label("", id="input-form-time-error", classes="input-form-error")

            with Vertical(id="input-form-n", classes="input-form-input"):
                yield Label(
                    "Trotterisation level: ",
                    id="input-form-n-label",
                    classes="input-form-label",
                )
                yield Input(
                    id="input-form-n-input",
                    classes="input-form-input-component",
                    placeholder="Simulation accuracy/trotter terms",
                    validators=[Integer(minimum=1)],
                    validate_on=["blur", "submitted"],
                )
                yield Label("", id="input-form-n-error", classes="input-form-error")

            with Vertical(id="input-form-kinds", classes="input-form-input"):
                yield Label(
                    "Simulation kind(s): ",
                    id="input-form-kinds-label",
                    classes="input-form-label",
                )
                yield SelectionList(*self.SELECTIONS, id="input-form-kinds-input")
                yield Label("", id="input-form-kinds-error", classes="input-form-error")

            with Horizontal(id="input-form-button-area"):
                yield Button(
                    "Run!",
                    variant="primary",
                    id="input-form-submit-button",
                    disabled=True,
                )

    ##
    # Event handlers
    ##

    @on(DescendantBlur, ".input-form-input-component")
    @on(Input.Submitted)
    def handle_input_change(self, event: Blur):
        control = cast(Input, event.control)
        assert control is not None
        validation = control.validate(control.value)
        if validation is None:
            self.notify(
                "Validation gave None when it should not have!", severity="warning"
            )
            return

        assert control.id is not None

        self.update_input_value(
            control.id, control.value if validation.is_valid else None
        )

        if control.has_parent:
            parent = control.parent
            assert parent is not None
            err_label = parent.query_one(".input-form-error", Label)
            err_label.update(
                "\n".join(validation.failure_descriptions)
                if not validation.is_valid
                else ""
            )

        self.update_button_state()

    @on(SelectionList.SelectedChanged)
    def handle_selection_change(self, event: SelectionList.SelectedChanged):
        control = event.control
        replacement = SimulationKindSet(0)
        for el in control.selected:
            assert isinstance(el, SimulationKindSet)
            replacement = replacement | el
        self.kind_selection_input = replacement
        self.update_button_state()

    @on(Button.Pressed)
    def handle_button(self, _: Button.Pressed):
        fail = self.update_button_state()
        if fail:
            self.notify(
                "Button was pressed but should have been disabled", severity="warning"
            )
            return

        messager = self.FormSubmitted(
            pauli=cast(str, self.pauli_input),
            time=cast(float, self.time_input),
            n=cast(int, self.n_input),
            kind=self.kind_selection_input,
        )

        self.post_message(messager)

    ##
    # Helper methods
    ##

    def update_input_value(self, id: str, value: str | None):
        if id == "input-form-pauli-input":
            self.pauli_input = value
        if id == "input-form-n-input":
            try:
                self.n_input = int(value) if value is not None else None
            except ValueError:
                self.notify(
                    "ValueError was raised while saving simulation accuracy!",
                    severity="error",
                )
        if id == "input-form-time-input":
            try:
                self.time_input = float(value) if value is not None else None
            except ValueError:
                self.notify(
                    "ValueError was raised while saving time steps!",
                    severity="error",
                )

    def update_button_state(self) -> bool:
        """[TODO:description]

        :return: a boolean for if the button should and has been disabled or not
        """
        inputs = [self.pauli_input, self.n_input, self.time_input]
        any_empty = any([x is None for x in inputs])

        disabled = any_empty or len(self.kind_selection_input) == 0
        self.query_one(Button).disabled = disabled
        return disabled

    class FormSubmitted(Message):
        def __init__(
            self, pauli: str, time: float, n: int, kind: SimulationKindSet
        ) -> None:
            super().__init__()
            self.pauli = pauli
            self.time = time
            self.n = n
            self.kind = kind
