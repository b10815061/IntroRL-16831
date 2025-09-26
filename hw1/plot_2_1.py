import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# --- Helper to load scalar values from TensorBoard logs ---
def load_tensorboard_scalars(path, tag="Eval_AverageReturn"):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    if tag not in ea.scalars.Keys():
        print(f"Available tags in {path}: {ea.scalars.Keys()}")
        raise KeyError(f"Tag '{tag}' not found in {path}")
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    std = ea.Scalars('Eval_StdReturn')
    stds = [e.value for e in std]
    return np.array(steps), np.array(values), np.array(stds)

# --- Paths to your log files ---
log_ant = "./submission/dagger_ant/events.out.tfevents.1758060247.ff2ac7f69338"
log_other = "./submission/dagger_hopper/events.out.tfevents.1758060481.ff2ac7f69338"

# --- Load curves ---
steps_ant, returns_ant,stds_ant = load_tensorboard_scalars(log_ant)
steps_other, returns_other,stds_other = load_tensorboard_scalars(log_other)

# --- Dummy baselines (replace with actual values from your runs) ---
expert_return_ant = 4713.65   # Example
bc_return_ant = 1462.01         # Example
expert_return_other = 3772.67
bc_return_other = 215.10

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Ant plot
axes[0].errorbar(
    steps_ant,
    returns_ant,
    yerr=stds_ant,
    fmt='-o',
    capsize=5,
    label="DAgger Policy",
    color="C0"
)
axes[0].axhline(expert_return_ant, color="C2", linestyle="--", label="Expert")
axes[0].axhline(bc_return_ant, color="C1", linestyle=":", label="Behavioral Cloning")
axes[0].set_title("Ant-v2")
axes[0].set_xlabel("DAgger Iterations")
axes[0].set_ylabel("Mean Return")
axes[0].legend()

# Other environment plot
axes[1].errorbar(
    steps_other,
    returns_other,
    yerr=stds_other,
    fmt='-o',
    capsize=5,
    label="DAgger Policy",
    color="C0"
)
axes[1].axhline(expert_return_other, color="C2", linestyle="--", label="Expert")
axes[1].axhline(bc_return_other, color="C1", linestyle=":", label="Behavioral Cloning")
axes[1].set_title("Hopper-v2")
axes[1].set_xlabel("DAgger Iterations")

# Set scales for both plots
max_return = max(expert_return_ant, returns_ant.max(), expert_return_other, returns_other.max())
axes[0].set_ylim(0, max_return * 1.1)
axes[1].set_ylim(0, max_return * 1.1)


plt.tight_layout()
# plt.show()
plt.savefig('Dagger_Ant_vs_Hopper.png')