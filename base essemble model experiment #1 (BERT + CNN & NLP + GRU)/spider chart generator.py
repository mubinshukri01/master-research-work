import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Excel file
file_path = "Mubin Master Finalize experiment.xlsx"

# Read performance sheet
df = pd.read_excel(file_path, sheet_name="Performance optimized ensembled")

# Clean dataframe
df = df.dropna(how='all')
df = df.dropna(axis=1, how='all')

# Example structure
# Model | Accuracy | Precision | Recall | F1-Macro
models = df.iloc[:,0]
metrics = df.columns[1:]

num_metrics = len(metrics)

angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

plt.figure(figsize=(7,7))
ax = plt.subplot(111, polar=True)

for i in range(len(models)):
    
    values = df.iloc[i,1:].values
    values = np.concatenate((values, [values[0]]))
    
    ax.plot(angles, values, linewidth=2, label=models[i])
    ax.fill(angles, values, alpha=0.1)

ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics)

plt.title("Model Performance Comparison")
plt.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))

plt.tight_layout()
plt.show()