import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# --- Helper to load scalar values from TensorBoard logs ---
def load_tensorboard_scalars(path, tag="Eval_AverageReturn"):
    """Loads scalars from a single tfevents file."""
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    if tag not in ea.scalars.Keys() or 'Eval_StdReturn' not in ea.scalars.Keys():
        print(f"Available tags in {path}: {ea.scalars.Keys()}")
        raise KeyError(f"Required tag '{tag}' or 'Eval_StdReturn' not found in {path}")
        
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    std_events = ea.Scalars('Eval_StdReturn')
    stds = [e.value for e in std_events]
    
    return np.array(steps), np.array(values), np.array(stds)

# --- Paths to your log file directories ---
# This finds directories like './submission/q1_test_ant_32'
log_dirs = glob.glob('./submission/q1_test*')

eval_avgs = []
eval_stds = []
b_s = []

for log_dir in log_dirs:
    # Extract the batch size from the directory name
    try:
        train_batch_size = int(log_dir.split('_')[-1])
    except (ValueError, IndexError):
        print(f"Could not parse batch size from directory name: {log_dir}. Skipping.")
        continue

    # Find the tfevents file inside the directory
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        print(f"No event file found in {log_dir}. Skipping.")
        continue
    
    # Load data from the first event file found
    event_file_path = event_files[0]
    _, avg, std = load_tensorboard_scalars(event_file_path)
    
    # --- FIX: Append only the FINAL value from each run ---
    if avg.size > 0 and std.size > 0:
        eval_avgs.append(avg[-1])
        eval_stds.append(std[-1])
        b_s.append(train_batch_size)
    else:
        print(f"No data found in {event_file_path}. Skipping.")

# Sort the data by batch size for a clean plot
sorted_indices = np.argsort(b_s)
b_s = np.array(b_s)[sorted_indices]
eval_avgs = np.array(eval_avgs)[sorted_indices]
eval_stds = np.array(eval_stds)[sorted_indices]

# --- Plot ---
fig, axes = plt.subplots(1, 1, figsize=(10, 6))

axes.errorbar(
    b_s,
    eval_avgs,
    yerr=eval_stds,
    fmt='-o',
    capsize=5,
    label="Final Mean Return",
    color="C0"
)
axes.set_title("Final Performance vs. Batch Size on Ant-v2")
axes.set_xlabel("Training Batch Size") # <-- FIX: Correct label
axes.set_ylabel("Mean Return")
axes.legend()

# Set scales for the plot
if eval_avgs.size > 0:
    max_return = (eval_avgs + eval_stds).max()
    axes.set_ylim(0, max_return * 1.1)
    axes.set_xticks([0,1000,2000,4000])

plt.tight_layout()
plt.savefig('batch_size_vs_return.png')
print("Plot saved to batch_size_vs_return.png")