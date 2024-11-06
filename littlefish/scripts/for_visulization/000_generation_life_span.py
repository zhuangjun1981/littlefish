import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

gen_folder = r"F:\little_fish_simulation_logs_4\generation_0000000"

names = []
life_spans = []
clip_top_life_span = 30000

for fn in os.listdir(gen_folder):
    if fn[0:5] == "fish_" and fn[-5:] == ".hdf5":
        curr_f = h5py.File(os.path.join(gen_folder, fn), "r")
        sim_n = [s for s in curr_f.keys() if s[:11] == "simulation_"]
        if len(sim_n) != 1:
            raise ValueError(
                "number of simulation for fish ({}) does not "
                "equal one".format(curr_f["fish/name"][()])
            )
        sim_n = sim_n[0]
        names.append(fn)
        life_spans.append(curr_f[sim_n]["simulation_log/last_time_point"][()])
        curr_f.close()

life_span_df = pd.DataFrame()
life_span_df["life_span"] = life_spans
life_span_df["name"] = names
life_span_df.sort_values(by="life_span", inplace=True)
print(life_span_df)

life_spans_plot = np.clip(life_spans, 0, clip_top_life_span)
values, bin_edges = np.histogram(
    life_spans_plot, bins=60, range=[0, clip_top_life_span]
)
bin_width = np.mean(np.diff(bin_edges))
bin_centers = (bin_edges[:-1] + bin_width / 2.0).astype(np.int32)

df_plot = pd.DataFrame(
    {
        "fish count": values,
        "life span": bin_centers,
    }
)

# print(df_plot)
generation = int(os.path.split(gen_folder)[-1].split("_")[-1])
mean_life_span = np.mean(life_spans)
# median_life_span = np.median(life_spans)

f, ax = plt.subplots(figsize=(7, 5))
ax.axvline(x=10000, ls="--", color="r")
ax.bar(df_plot["life span"], df_plot["fish count"], width=bin_width)
ax.set_xlabel("life span", fontsize=16)
ax.set_ylabel("fish count", fontsize=16)
ax.set_title(
    f"generation: {generation:05d}, mean life span: {mean_life_span:.0f}", fontsize=16
)
# ax.set_title(
#     f"generation: {generation:05d}, median life span: {median_life_span:.0f}", fontsize=16
# )
# plt.tight_layout()
plt.show()
