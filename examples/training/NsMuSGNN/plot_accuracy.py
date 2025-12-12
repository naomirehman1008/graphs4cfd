import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")

data = pd.read_csv("physics_informed_accuracy.csv")

def process_scales(row):
    name = row["model_name"]
    if "One" in name:
        return 'OneScale'
    elif "Two" in name:
        return 'TwoScale'
    elif "Three" in name:
        return 'ThreeScale'
    elif "Four" in name:
        return 'FourScale'
    else:
        return None  # or np.nan

data["scales"] = data.apply(process_scales, axis=1)
    
def process_config(row):
    if row["div"] == 0:
        return "Baseline"
    elif row["div"] == 1 and row["mom"] == 0.1 and row["spec"] == 0.05:
        return "Div. + Mom. + Spec."
    elif row["div"] == 1 and row["mom"] == 0.1:
        return "Div. and Mom."
    elif row["div"] == 1:
        return "Div."
    else:
        return None  # or some other label

data["config"] = data.apply(process_config, axis=1)

fig, axes = plt.subplots(1, 4, figsize=(12, 4))

def bar_chart(data_in, dataset, index):
    # Draw a nested barplot by species and sex
    sns.barplot(
        data=data_in,
        x="scales", y="accuracy", hue="config", palette="Paired", alpha=1.0, ax=axes[index]
    )
    axes[index].set_ylabel("Coef. of Det.")
    axes[index].set_xlabel(dataset)
    axes[index].set_xticklabels(axes[index].get_xticklabels(), rotation=37, ha='center', size=11)

filtered_data = data[data['dataset'] == 'ns_circle']
bar_chart(filtered_data, 'Train', 0)
filtered_data = data[data['dataset'] == 'ns_circle_low_re']
bar_chart(filtered_data, 'Low Re', 1)
filtered_data = data[data['dataset'] == 'ns_circle_mid_re']
bar_chart(filtered_data, 'Mid Re', 2)
filtered_data = data[data['dataset'] == 'ns_circle_high_re']
bar_chart(filtered_data, 'High Re', 3)

for ax in axes:
    ax.legend_.remove()

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(labels))
plt.tight_layout()
plt.subplots_adjust(bottom=0.35) 

fig.savefig(f"plot.pdf", dpi=300, format='pdf')