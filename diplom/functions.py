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
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM


def generate_spectre(coeff: [torch.Tensor, tuple] = None) -> Tuple[np.ndarray, Tuple[float, float, float, float, float, float]]:

    W = np.linspace(0.05, -0.7, num=256)
    K = np.linspace(-0.4, 0.4, num=256)
    k, w = np.meshgrid(K, W)

    if coeff is None:
        alpha = random.uniform(2, 10)
        imp = random.uniform(0.01, 1)
        m = random.uniform(0, 10)
        l = random.uniform(0, 1)
        lmb = random.uniform(0, 10)
        shirley = random.uniform(0.1, 5)
    else:
        if isinstance(coeff, torch.Tensor):
            coeff = coeff.tolist()
        alpha = coeff[0]
        imp = coeff[1]
        m = coeff[2]
        l = coeff[3]
        lmb = coeff[4]
        shirley = coeff[5]


    param = alpha * (w ** 2) + imp
    result = param / (((1 - lmb) * w - (m * k ** 2 + l)) ** 2 + param ** 2)


    # Shirli
    shirley_noise = shirley * w ** 2

    result += shirley_noise

    # Gause noise
    noise = np.random.normal(loc=0.0, scale=0.05, size=result.shape)

    spectra_noise = result + ((noise / 4) * result)
    spectra_noise = (spectra_noise - np.min(spectra_noise)) / (np.max(spectra_noise) - np.min(spectra_noise))

    # Fermi level
    kb = 8.617 * (10 ** (-5))
    T = random.uniform(0, 250)
    Ef = 0
    exp = 1 / (np.exp((w + Ef) / (kb * T)) + 1)

    result = spectra_noise * exp

    return np.expand_dims(result, axis=0), (alpha, imp, m, l, lmb, shirley)


def export_data(figure: Figure):
    figure.write_html(f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.html")


def show_diff(real: List[Tuple[np.ndarray, Tuple[float, float, float, float, float, float]]],
              pred: List[Tuple[np.ndarray, Tuple[float, float, float, float, float, float]]], lv=None) -> None:
    figure = make_subplots(rows=1, cols=2)

    coeff = real[1]
    figure.add_trace(go.Heatmap(z=real[0][0], showscale=False,
                                colorscale='Greys_r',  # Colormap
                                colorbar=dict(
                                    title='Intensity',
                                    thickness=20,  # толщина шкалы
                                    len=0.75,  # длина шкалы (0..1)
                                    x=1.02
                                )
                                ), row=1, col=1)
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
    figure.add_trace(go.Heatmap(z=pred[0][0], showscale=False,
                                colorscale='Greys_r',  # Colormap
                                colorbar=dict(
                                    title='Intensity',
                                    thickness=20,  # толщина шкалы
                                    len=0.75,  # длина шкалы (0..1)
                                    x=1.02
                                )
                                ), row=1, col=2)
    figure.add_annotation(
        text=f"alpha={coeff[0]}<br>imp={coeff[1]}<br>m={coeff[2]}<br>l={coeff[3]}<br>lambda={coeff[4]}<br>Pred",
        xref="x domain",
        yref="y domain",
        col=2,
        row=1,
        y=-0.3,
        showarrow=False
    )

    ssim = SSIM(data_range=1.0, kernel_size=5, sigma=0.5)
    similarity = ssim(torch.stack([torch.tensor(real[0])]), torch.stack([torch.tensor(pred[0])]))

    figure.update_layout(
        height=600,
        margin=dict(t=50, b=150),
        title_text=f"similarity = {similarity}" if lv is None else f"similarity = {similarity} LV{lv}K"
    )

    figure.show()


def show_spectre(inp: Tuple[np.ndarray, Tuple[float, float, float, float, float, float]]) -> plt:

    plt.imshow(inp[0].squeeze(), cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Intensity")  # Добавляем цветовую шкалу
    plt.title("Heatmap using imshow()")

    return plt


def show_data(data: List[Tuple[np.ndarray, Tuple[float, float, float, float, float, float]]], num_cols=5, title=None) -> Figure:
    figure = make_subplots(rows=len(data) // num_cols, cols=num_cols)

    for i, d in enumerate(data):
        coeff = d[1]
        row = (i // num_cols) + 1
        col = (i % num_cols) + 1

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
        height=450 * len(data) // num_cols,
        margin=dict(t=50, b=100)
    )

    if title is not None:
        figure.update_layout(
            title_text=f"{title}"
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
    values = [generate_spectre() for _ in range(5000)]


def get_model_summary(modell, input_size):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        summary(modell, input_size=input_size)
    return stream.getvalue()


if __name__ == "__main__":
    # test_spectre, params = generate_spectre()
    # recreated_spectre, params = generate_spectre(params)
    # show_diff((test_spectre, params), (recreated_spectre, params))
    show_data([generate_spectre() for _ in range(5)], title="square")