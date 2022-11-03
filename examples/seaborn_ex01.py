import matplotlib.pyplot as plt
import seaborn as sns

from mpl_selector import Selector

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

grouped_selector.select("").difference("", category="Sat").set("alpha", 0.5)

plt.show()

