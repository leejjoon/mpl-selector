import matplotlib
matplotlib.use("Agg")

import warnings
from mpl_selector import Selector

def test():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import seaborn_boxplot

    ax = seaborn_boxplot.ax

    selector = Selector(ax)
    grouped_selector = selector.guess_categorical(axis="x")
    selectee = grouped_selector.select("Path").difference(category="Sat")

    assert selectee.indices == set([1, 3, 4, 5, 8, 9])

