from textual import on
from textual.binding import Binding
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import ContentSwitcher, Footer, Header, Tab, Tabs
from hamil_clever_sim.hamil_runner import SimulationRunner

from hamil_clever_sim.inputs import InputForm
from hamil_clever_sim.simulation_result import SimulationDisplay


class MainView(App):
    CSS_PATH = "./main.tcss"

    BINDINGS = [
        Binding(
            "ctrl+q",
            "quit",
            "Quit simulator",
            priority=True
        )
    ]

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="app-main"):
            with Vertical(id="app-main-input"):
                yield InputForm()

            with Vertical(id="app-main-display"):
                yield Tabs(id="app-main-display-tabs")
                yield ContentSwitcher(id="app-main-display-switcher")

        yield Footer()

    @on(InputForm.FormSubmitted)
    async def handle_simulation_run(self, event: InputForm.FormSubmitted) -> None:
        switcher = self.query_one(
            "#app-main-display-switcher", ContentSwitcher)
        tabber = self.query_one("#app-main-display-tabs", Tabs)

        id = f"simulation-{len(switcher.children)}"
        tab = Tab(id, id=id)

        runner = SimulationRunner(event.pauli, event.time, event.n, event.kind)
        displayer = SimulationDisplay(id=id)

        displayer.loading = True
        await switcher.mount(displayer)
        tabber.add_tab(tab)
        tabber.active = id
        tabber.show(id)

        displayer.runner = runner

    @on(Tabs.TabActivated)
    def handle_tab_active(self, event: Tabs.TabActivated) -> None:
        switcher = self.query_one(
            "#app-main-display-switcher", ContentSwitcher)
        switcher.current = event.tab.id
