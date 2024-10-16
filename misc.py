import time
import matplotlib as mpl
import matplotlib.pyplot as plt


def timer(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("\nTotal training time:", end - start, "sec")
        return result

    return wrapper


def plot_settings(figsize=(20, 10)):
    """
    Prepares plot configurations.
    :return: plt args
    """
    # Seems that the first figure has to be discarded
    _ = plt.figure()

    mpl.rcParams['figure.figsize'] = figsize
    mpl.rcParams["figure.titlesize"] = 35
    mpl.rcParams['lines.markersize'] = 10
    mpl.rcParams['axes.titlesize'] = 26
    mpl.rcParams['axes.labelsize'] = 26
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20
    fig = plt.figure(constrained_layout=True)
    return fig
