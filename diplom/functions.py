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
import torch
import matplotlib.pyplot as plt


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


def generate_spectre_by_values(coeff: torch.Tensor):
    coeff = coeff.tolist()

    W = np.linspace(0, -0.7, num=256)
    K = np.linspace(-0.4, 0.4, num=256)
    k, w = np.meshgrid(K, W)
    alpha = coeff[0]
    imp = coeff[1]
    m = coeff[2]
    l = coeff[3]
    lmb = coeff[4]

    param = alpha * (w ** 2) + imp
    result = param / (((1 - lmb) * w - (m * k ** 2 + l)) ** 2 + param ** 2)

    return np.expand_dims(result, axis=0), (alpha, imp, m, l, lmb)


def export_data(figure: Figure):
    figure.write_html(f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.html")


def show_diff(real: List[Tuple[np.ndarray, Tuple[float, float, float, float, float]]],
              pred: List[Tuple[np.ndarray, Tuple[float, float, float, float, float]]]) -> None:
    figure = make_subplots(rows=1, cols=2)

    coeff = real[1]
    figure.add_trace(go.Heatmap(z=real[0][0], showscale=False), row=1, col=1)
    figure.add_annotation(
        text=f"alpha={coeff[0]}<br>imp={coeff[1]}<br>m={coeff[2]}<br>l={coeff[3]}<br>lambda={coeff[4]}<br>Real",
        xref="x domain",
        yref="y domain",
        col=1,
        row=1,
        y=-0.3,
        showarrow=False
    )

    coeff = pred[1]
    figure.add_trace(go.Heatmap(z=pred[0][0], showscale=False), row=1, col=2)
    figure.add_annotation(
        text=f"alpha={coeff[0]}<br>imp={coeff[1]}<br>m={coeff[2]}<br>l={coeff[3]}<br>lambda={coeff[4]}<br>Pred",
        xref="x domain",
        yref="y domain",
        col=2,
        row=1,
        y=-0.3,
        showarrow=False
    )
    figure.update_layout(
        height=700,
        margin=dict(t=50, b=150)
    )

    figure.show()


def show_spectre(inp: Tuple[np.ndarray, Tuple[float, float, float, float, float]]) -> plt:

    plt.imshow(inp[0].squeeze(), cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Intensity")  # Добавляем цветовую шкалу
    plt.title("Heatmap using imshow()")

    return plt


def show_data(data: List[Tuple[np.ndarray, Tuple[float, float, float, float, float]]]) -> Figure:
    figure = make_subplots(rows=len(data) // 5, cols=5)

    for i, d in enumerate(data):
        coeff = d[1]
        row = (i // 5) + 1
        col = (i % 5) + 1

        figure.add_trace(go.Heatmap(z=d[0][0], showscale=False, colorscale=[[0, 'rgb(0,0,0)'], [1,'rgb(255,255,255)']]),
                         row=row, col=col)
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
