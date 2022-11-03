import matplotlib.pyplot as plt
import seaborn as sns

from mpl_visual_context.patheffects import HLSModifyStroke

df_peng = sns.load_dataset("penguins")

fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True, clear=True)
sns.countplot(y="species", data=df_peng, ax=ax)

from mpl_selector import Selector
selector = Selector(ax)
grouped_selector = selector.guess_categorical(axis="y",
                                              ignore_columns=["fc"])

grouped_selector.select("").difference("", category="Gentoo").set("alpha", 0.2)

plt.show()

