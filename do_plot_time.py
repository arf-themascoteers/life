import plotter_time


def plot(*args, **kwargs):
    plotter_time.plot(*args, **kwargs)


plot(only_algorithms=["bsdr","mcuve"])