import matplotlib.pyplot as plt
import seaborn as sns
from mpl_selector import Selector

sns.set_context("talk")
sns.set_style("white")

df = sns.load_dataset("tips")

fig = plt.figure(figsize=(10, 3))
ax = fig.add_subplot(111)

sns.violinplot(data=df, x="day", y="total_bill", ax=ax,
               hue="sex",
               split=True,
               linewidth=1
               )

selector = Selector(ax)
grouped = selector.guess_categorical(axis="x",
                                     import_legend=True)

markers = grouped.select("PathCollection")
markers.set("sizes", [8**2])

box = grouped.select("Line2D", lw=3.)

# While we can do above to select 'box', the code will break if we change
# linewidth when violinplot is created.

# Instead, we groupby the artists by their 'lw' values, assuming the groups
# with larger lw corresponds to the 'box'.

_, box = grouped.select("Line2D").groupby("lw", values_only=True)

box.set("lw", box[0].get_lw() * 2)

grouped.select("").difference("", category="Sat").set("alpha", 0.5)

ax.get_legend().remove()
fig.legend(loc="upper right", ncol=2)

plt.show()
