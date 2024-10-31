import plotter_regression
import plotter_classification


def plot(*args, **kwargs):
    plotter_classification.plot_combined(*args, **kwargs)
    plotter_regression.plot_combined(*args, **kwargs)


plot(sources=["t1"], only_algorithms=["skipattn","stage3attn"])