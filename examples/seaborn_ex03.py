import matplotlib.pyplot as plt
import seaborn as sns
from mpl_selector import Selector

sns.set_theme(style="whitegrid")

diamonds = sns.load_dataset("diamonds")
clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

ax = sns.boxenplot(x="clarity", y="carat",
                   color="b", order=clarity_ranking,
                   scale="linear", data=diamonds)


selector = Selector(ax)
grouped_selector = selector.guess_categorical(axis="x")

grouped_selector.select("", category="VVS2").set("alpha", 0.2)
grouped_selector.select("Line").set("lw", 2).set("color", "y")

plt.show()
