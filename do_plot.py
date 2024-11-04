import plotter_regression
import plotter_classification


def plot(*args, **kwargs):
    plotter_classification.plot_combined(*args, **kwargs)
    plotter_regression.plot_combined(*args, **kwargs)


plot(sources=["p1","p3","p5","p6","p8"],only_algorithms=["bsdr","c1","bsdrattn2"])