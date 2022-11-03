import matplotlib.pyplot as plt
import seaborn as sns
from mpl_selector import Selector, groupby
from mpl_visual_context.patheffects import HLSModifyStroke

sns.set_theme(style="whitegrid")

diamonds = sns.load_dataset("diamonds")
clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

ax = sns.boxenplot(x="clarity", y="carat",
                   color="b", order=clarity_ranking,
                   scale="linear", data=diamonds)


selector = Selector(ax)
grouped_selector = selector.guess_categorical(axis="x")
print(grouped_selector.group_keys)

pe = [HLSModifyStroke(dh=0.25)]

for a in grouped_selector.select("PatchColl", category="VVS2"):
    a.set_path_effects(pe)

for a in grouped_selector.select("Line"):
    a.set_lw("2")
    a.set_color("y")

plt.show()
