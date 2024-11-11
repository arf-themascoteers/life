import plotter_regression
import plotter_classification


def plot(*args, **kwargs):
    plotter_classification.plot_combined(*args, **kwargs)
    plotter_regression.plot_combined(*args, **kwargs)


plotter_classification.plot_combined(sources=["p21","p22"],only_algorithms=["bsnet","bnc","c1","c2"], only_datasets=["ghisaconus_health"])