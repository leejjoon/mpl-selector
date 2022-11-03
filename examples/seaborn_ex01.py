import matplotlib.pyplot as plt
import seaborn as sns

from mpl_selector import Selector, groupby
from mpl_visual_context.patheffects import HLSModifyStroke

sns.set_theme(style="whitegrid")

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested violinplot and split the violins for easier comparison
ax = sns.violinplot(data=tips, x="day", y="total_bill", hue="smoker",
                    split=True, inner="quart", linewidth=1,
                    palette={"Yes": "b", "No": ".85"})
sns.despine(left=True)

selector = Selector(ax)
grouped_selector = selector.guess_categorical(axis="x",
                                              import_legend=True)

pe = [HLSModifyStroke(ds=-0.2, l="-60%")]

for a in grouped_selector.select("", label="Yes"):
    a.set_path_effects(pe)
for a in grouped_selector.select("", label="No"):
    a.set_path_effects(pe)

for a in grouped_selector.select("", category="Sat"):
    a.set_path_effects(None)

# for a in grouped_selector.select("Line"):
#     a.set_lw(2)

print(grouped_selector.group_keys)

plt.show()

