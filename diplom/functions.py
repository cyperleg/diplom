import time
import numpy as np
import random
import plotly.graph_objs as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from typing import List, Tuple
import datetime
import io
import contextlib
from torchsummary import summary


def generate_spectre() -> Tuple[np.ndarray, Tuple[float, float, float, float, float]]:
    W = np.linspace(0, -0.7, num=256)
    K = np.linspace(-0.4, 0.4, num=256)
    k, w = np.meshgrid(K, W)
    alpha = random.uniform(2, 15)
    imp = random.uniform(1, 2)
    m = random.uniform(0, 10)
    l = random.uniform(0, 1)
    lmb = random.uniform(0, 10)

    param = alpha * (w ** 2) + imp
    result = param / (((1 - lmb) * w - (m * k ** 2 + l)) ** 2 + param ** 2)

    return np.expand_dims(result, axis=0), (alpha, imp, m, l, lmb)


def export_data(figure: Figure):
    figure.write_html(f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.html")


def show_data(data: List[Tuple[np.ndarray, Tuple[float, float, float, float, float]]]) -> Figure:
    figure = make_subplots(rows=len(data) // 5, cols=5)

    for i, d in enumerate(data):
        coeff = d[1]
        row = (i // 5) + 1
        col = (i % 5) + 1

        figure.add_trace(go.Heatmap(z=d[0][0], showscale=False), row=row, col=col)
        figure.add_annotation(
            text=f"alpha={coeff[0]}<br>imp={coeff[1]}<br>m={coeff[2]}<br>l={coeff[3]}<br>lambda={coeff[4]}",
            xref="x domain",
            yref="y domain",
            col=col,
            row=row,
            y=-0.3,
            showarrow=False
        )

    figure.update_layout(
        height=450 * len(data) // 5,
        margin=dict(t=50, b=100)
    )

    figure.show()

    return figure


def time_tracker(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Начало отслеживания времени
        result = func(*args, **kwargs)     # Вызов функции
        end_time = time.perf_counter()      # Конец отслеживания времени
        duration = end_time - start_time    # Расчет продолжительности
        print(f"Время выполнения '{func.__name__}': {duration:.4f} секунд")
        return result                       # Возвращаем результат функции
    return wrapper


@time_tracker
def check():
    values = [generate_spectre() for _ in range(50000)]


def get_model_summary(modell, input_size):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        summary(modell, input_size=input_size)
    return stream.getvalue()


if __name__ == "__main__":
    value = [generate_spectre() for _ in range(10)]
    show_data(value)
