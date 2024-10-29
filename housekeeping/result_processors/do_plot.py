import plotter_regression
import plotter_classification


def plot(*args, **kwargs):
    plotter_classification.plot_combined(*args, **kwargs)
    plotter_regression.plot_combined(*args, **kwargs)


#plot(only_algorithms=["bsdrattn","skipattn","lateattn","stage3attn"])
#plot(only_algorithms=["bsdrattn","lateattn"])
#plot(only_algorithms=["lateattn","stage3attn"])
#plot(only_algorithms=["bsdrattn","stage3attn"])
#plot(["."],only_algorithms=["bsdrattn","skipattn"], pending=True)
plot(only_algorithms=["bsdrattn","skipattn"])