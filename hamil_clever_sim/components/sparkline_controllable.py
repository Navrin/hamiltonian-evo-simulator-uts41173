from __future__ import annotations

from fractions import Fraction
from typing import Callable, ClassVar, Generic, Iterable, Optional, Sequence, TypeVar

from rich.color import Color
from rich.console import Console, ConsoleOptions, RenderResult as RichRenderResult
from rich.segment import Segment
from rich.style import Style
from textual.app import RenderResult
from textual.reactive import Reactive, reactive

from textual.renderables._blend_colors import blend_colors
from textual.widget import Widget

T = TypeVar("T", int, float)

SummaryFunction = Callable[[Sequence[T]], float]


class SparklinePrimative(Generic[T]):
    BARS = "▁▂▃▄▅▆▇█"

    def __init__(
        self,
        data: Sequence[T],
        *,
        width: int | None,
        marked_index: int = -1,
        min_color: Color = Color.from_rgb(0, 255, 0),
        max_color: Color = Color.from_rgb(255, 0, 0),
        summary_function: SummaryFunction[T] = max,
    ) -> None:
        self.data: Sequence[T] = data
        self.width = width
        self.min_color = Style.from_color(min_color)
        self.max_color = Style.from_color(max_color)
        self.summary_function: SummaryFunction[T] = summary_function
        self.marked_index = marked_index

    @classmethod
    def _buckets(cls, data: Sequence[T], num_buckets: int) -> Iterable[Sequence[T]]:
        bucket_step = Fraction(len(data), num_buckets)
        for bucket_no in range(num_buckets):
            start = int(bucket_step * bucket_no)
            end = int(bucket_step * (bucket_no + 1))
            partition = data[start:end]
            if partition:
                yield partition

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RichRenderResult:
        width = self.width or options.max_width
        len_data = len(self.data)
        if len_data == 0:
            yield Segment("▁" * width, self.min_color)
            return
        if len_data == 1:
            yield Segment("█" * width, self.max_color)
            return

        minimum, maximum = min(self.data), max(self.data)
        extent = maximum - minimum or 1

        buckets = tuple(self._buckets(self.data, num_buckets=width))

        bucket_index = 0.0
        bars_rendered = 0
        step = len(buckets) / width
        summary_function = self.summary_function
        min_color, max_color = self.min_color.color, self.max_color.color
        assert min_color is not None
        assert max_color is not None
        while bars_rendered < width:
            partition = buckets[int(bucket_index)]
            partition_summary = summary_function(partition)
            height_ratio = (partition_summary - minimum) / extent
            bar_index = int(height_ratio * (len(self.BARS) - 1))
            bar_color = blend_colors(min_color, max_color, height_ratio)
            bars_rendered += 1
            bucket_index += step
            if partition == buckets[self.marked_index]:
                yield Segment(
                    self.BARS[bar_index], Style.from_color(Color.from_rgb(230, 30, 160))
                )
            else:
                yield Segment(self.BARS[bar_index], Style.from_color(bar_color))


def _max_factory() -> Callable[[Sequence[float]], float]:
    return max


class Sparkline(Widget):
    COMPONENT_CLASSES: ClassVar[set[str]] = {
        "sparkline--max-color",
        "sparkline--min-color",
    }
    DEFAULT_CSS = """
    Sparkline {
        height: 1;
    }
    Sparkline > .sparkline--max-color {
        color: $accent;
    }
    Sparkline > .sparkline--min-color {
        color: $accent 30%;
    }
    """

    data = reactive[Optional[Sequence[float]]](None)
    summary_function = reactive[Callable[[Sequence[float]], float]](_max_factory)
    marked_index = reactive(-1)

    def __init__(
        self,
        data: Sequence[float] | None = None,
        *,
        summary_function: Callable[[Sequence[float]], float] | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.data = data
        if summary_function is not None:
            self.summary_function = summary_function

    def watch_marked_index(self, update: int):
        self.marked_index = update

    def render(self) -> RenderResult:
        if not self.data:
            return "<empty sparkline>"
        _, base = self.background_colors
        return SparklinePrimative(
            self.data,
            width=self.size.width,
            marked_index=self.marked_index,
            min_color=(
                base + self.get_component_styles("sparkline--min-color").color
            ).rich_color,
            max_color=(
                base + self.get_component_styles("sparkline--max-color").color
            ).rich_color,
            summary_function=self.summary_function,
        )
