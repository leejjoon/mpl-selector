import matplotlib.pyplot as plt
import seaborn as sns
from mpl_selector import Selector, groupby

sns.set_theme(style="whitegrid")

penguins = sns.load_dataset("penguins")

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=penguins, kind="bar",
    x="species", y="body_mass_g", hue="sex",
    errorbar="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Body mass (g)")
g.legend.set_title("")

selector = Selector(g.ax)
grouped_selector = selector.guess_categorical(axis="x")
grouped_selector.import_legend(g.legend)

grouped_selector.select("Rect").difference("", category="Gentoo").set("alpha", 0.5)

plt.show()
