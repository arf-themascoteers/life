import plotter_regression
import plotter_classification


def plot(*args, **kwargs):
    plotter_classification.plot_combined(*args, **kwargs)
    plotter_regression.plot_combined(*args, **kwargs)


plotter_classification.plot_combined(
    sources=["p21","p22","p23","p24"],
    only_algorithms=["c2","c3","c4"],
    only_datasets=["ghisaconus_health"]
)