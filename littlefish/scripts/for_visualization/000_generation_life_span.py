import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from littlefish.core import plotting

gen_folder = r"F:\little_fish_simulation_logs_3\generation_0000018"
max_life_span = 30000
bins = 60
default_life_span = 10000


life_span_df = plotting.get_geneartion_life_spans(gen_folder=gen_folder)

f, ax = plt.subplots(figsize=(7, 5))
ax.axvline(x=default_life_span, ls="--", color="r")

dist_df = plotting.plot_life_span_distribution(
    life_spans=life_span_df["life_span"],
    ax=ax,
    bins=bins,
    max_life_span=max_life_span,
)

generation = int(os.path.split(gen_folder)[-1].split("_")[-1])
mean_life_span = np.mean(life_span_df["life_span"])
ax.set_title(
    f"generation: {generation:05d}, mean life span: {mean_life_span:.0f}", fontsize=16
)

plt.tight_layout()
plt.show()
