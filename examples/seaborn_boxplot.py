import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="ticks", palette="pastel")

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested boxplot to show bills by day and time
ax = sns.boxplot(x="day", y="total_bill",
                 hue="smoker", palette=["m", "g"],
                 data=tips)
sns.despine(offset=10, trim=True)

from mpl_selector import Selector, groupby
from mpl_visual_context.patheffects import HLSModifyStroke

selector = Selector(ax)
grouped_selector = selector.guess_categorical(axis="x")

pe = [HLSModifyStroke(ds=-0.2, l="-50%")]

for a in grouped_selector.select(""):
    a.set_path_effects(pe)

for a in grouped_selector.select("", category="Sat"):
    a.set_path_effects(None)

plt.show()
