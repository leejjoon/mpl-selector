import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_context("talk")
sns.set_style("white")

df = sns.load_dataset("tips")

fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
# fig = plt.Figure()
# fig = plt.figure()
# ax = fig.add_subplot(111)

sns.violinplot(data=df, x="day", y="total_bill", ax=ax)

from mpl_selector import Selector, groupby
from mpl_visual_context.patheffects import HLSModifyStroke

selector = Selector(ax)
grouped_selector = selector.guess_categorical(axis="x",
                                              ignore_columns=["fc"])
# print(grouped_selector.group_keys)
pe = [HLSModifyStroke(ds=-0.2, l="-60%")]

for a in grouped_selector.select(""):
    a.set_path_effects(pe)
for a in grouped_selector.select("", category="Sat"):
    a.set_path_effects(None)

plt.show()
