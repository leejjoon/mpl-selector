import matplotlib.pyplot as plt
import seaborn as sns
from mpl_selector import Selector, groupby
from mpl_visual_context.patheffects import HLSModifyStroke

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

# pe1 = [ColorModifyStroke(ds=-0., dl=0.3)]
pe = [HLSModifyStroke(ds=-0.3, dl=0.3)]

for a in grouped_selector.select("Rect"):
    a.set_path_effects(pe)

for a in grouped_selector.select("Rect", category="Gentoo"):
    a.set_path_effects(None)

# print(grouped_selector.group_keys)

plt.show()
