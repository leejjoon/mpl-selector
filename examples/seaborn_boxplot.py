import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_selector import Selector

sns.set_theme(style="ticks", palette="pastel")

# Load the example tips dataset
tips = sns.load_dataset("tips")

fig = plt.figure()
ax = fig.add_subplot(111)

# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="day", y="total_bill",
            hue="smoker", palette=["m", "g"],
            data=tips, ax=ax)
sns.despine(offset=10, trim=True)

from mpl_visual_context.patheffects import HLSModifyStroke
pe = [HLSModifyStroke(s="50%", l="-50%")]

selector = Selector(ax)
grouped_selector = selector.guess_categorical(axis="x")
grouped_selector.select("").difference(category="Sat").set("path_effects", pe)

plt.show()
