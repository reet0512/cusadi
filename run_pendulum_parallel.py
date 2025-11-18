import os
import torch
import matplotlib.pyplot as plt
from casadi import *
from src import *

# === 1. Settings ===
BATCH_SIZE = 10000
N_PLOTS = 10                         # number of pendulums to log and plot
NUM_STEPS = 1000
CUSADI_FUNCTION_DIR = "src/casadi_functions"

# === 2. Load CasADi and CusADi functions ===
fn_casadi_sim_step = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "fn_sim_step.casadi"))
fn_cusadi_sim_step = CusadiFunction(fn_casadi_sim_step, BATCH_SIZE)

# === 3. Create random initial conditions (GPU tensors) ===
x0 = torch.rand((BATCH_SIZE, 2), device='cuda', dtype=torch.double)       # [theta, omega]
g  = 9.81 * torch.ones((BATCH_SIZE, 1), device='cuda', dtype=torch.double)
l  = torch.rand((BATCH_SIZE, 1), device='cuda', dtype=torch.double) + 0.5 # random lengths (avoid 0)
dt = torch.linspace(0.001, 0.1, BATCH_SIZE, device='cuda', dtype=torch.double)

print("Loaded CasADi function:", fn_casadi_sim_step)
print("CusADi function loaded successfully!")
print(f"Running multi-step simulation for {NUM_STEPS} steps...")

# === 4. Multi-step simulation ===
x = x0.clone()
theta_log = torch.zeros((NUM_STEPS, N_PLOTS), device='cuda', dtype=torch.double)

for step in range(NUM_STEPS):
    fn_cusadi_sim_step.evaluate([x, g, l, dt])
    x = fn_cusadi_sim_step.outputs_sparse[0].clone()
    theta_log[step] = x[:N_PLOTS, 0]   # record theta of first N_PLOTS pendulums

torch.cuda.synchronize()
print(f"✅ Simulation complete ({NUM_STEPS} steps × {BATCH_SIZE} pendulums).")

#print thetha logs for first 10 pendulums

print("Theta logs for first 10 pendulums:")
for i in range(N_PLOTS):
    print(f"Pendulum {i}: {theta_log[:, i].cpu().numpy()}")

# === 5. Plot and save results for first N_PLOTS pendulums ===
theta_cpu = theta_log.cpu().numpy()
fig, ax = plt.subplots(figsize=(9,5))
for i in range(N_PLOTS):
    ax.plot(theta_cpu[:, i], label=f'Pendulum {i}')
ax.set_xlabel("Time step")
ax.set_ylabel("Angle θ (radians)")
ax.set_title(f"Pendulum trajectories (first {N_PLOTS} pendulums, GPU parallel simulation)")
ax.legend(loc="upper right", ncol=2, fontsize=8)
ax.grid(True)
fig.tight_layout()

# save before showing
os.makedirs('results', exist_ok=True)
fig_path = os.path.join('results', f'pendulum_trajectories_first_{N_PLOTS}.png')
fig.savefig(fig_path, dpi=150)
print(f"✅ Figure saved to: {fig_path}")

plt.show()
print(f"✅ Trajectories for first {N_PLOTS} pendulums plotted.")

# import os
# import torch
# import matplotlib.pyplot as plt
# from casadi import *
# from src import *

# # === 1. Settings ===
# BATCH_SIZE = 10          # number of pendulums simulated in parallel
# N_PLOTS = 10                # number of pendulums to plot
# NUM_STEPS = 10
# CUSADI_FUNCTION_DIR = "src/casadi_functions"

# device = 'cuda'
# dtype = torch.double

# # === 2. Load CasADi and CusADi functions ===
# fn_casadi_sim_step = casadi.Function.load(
#     os.path.join(CUSADI_FUNCTION_DIR, "fn_sim_step.casadi")
# )
# fn_cusadi_sim_step = CusadiFunction(fn_casadi_sim_step, BATCH_SIZE)

# print("Loaded CasADi function:", fn_casadi_sim_step)
# print("CusADi function loaded successfully!")
# print(f"Running multi-step simulation for {NUM_STEPS} steps...")

# # === 3. Create random initial conditions (GPU tensors) ===
# # State: [theta, omega]
# x = torch.zeros((BATCH_SIZE, 2), device=device, dtype=dtype)
# x[:, 0] = torch.linspace(-0.9, 0.9, BATCH_SIZE, device=device, dtype=dtype)  # theta
# x[:, 1] = torch.zeros(BATCH_SIZE, device=device, dtype=dtype)                # omega

# g  = 9.81 * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)
# l  = 1.0 * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)
# dt = 0.01 * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)

# # === 4. Allocate log buffer ===
# theta_log = torch.zeros((NUM_STEPS, N_PLOTS), device=device, dtype=dtype)

# # === 5. Multi-step simulation ===
# for step in range(NUM_STEPS):
#     fn_cusadi_sim_step.evaluate([x, g, l, dt])
#     x = fn_cusadi_sim_step.outputs_sparse[0]
#     theta_log[step] = x[:N_PLOTS, 0]   # log theta of first N_PLOTS pendulums

# torch.cuda.synchronize()
# print(f"✅ Simulation complete ({NUM_STEPS} steps × {BATCH_SIZE} pendulums).")

# # === 6. Plot results ===
# theta_cpu = theta_log.cpu().numpy()

# fig, ax = plt.subplots(figsize=(9, 5))
# for i in range(N_PLOTS):
#     ax.plot(theta_cpu[:, i], label=f'Pendulum {i}')

# ax.set_xlabel("Time step")
# ax.set_ylabel("Angle θ (radians)")
# ax.set_title(f"Pendulum trajectories (first {N_PLOTS} pendulums, GPU parallel simulation)")
# ax.legend(loc="upper right", ncol=2, fontsize=8)
# ax.grid(True)
# fig.tight_layout()

# # Save before showing
# os.makedirs('results', exist_ok=True)
# fig_path = os.path.join('results', f'pendulum_trajectories_first_{N_PLOTS}.png')
# fig.savefig(fig_path, dpi=150)
# print(f"✅ Figure saved to: {fig_path}")

# plt.show()
# print(f"✅ Trajectories for first {N_PLOTS} pendulums plotted.")


# import os
# import torch
# from casadi import *
# from src import *

# CUSADI_FUNCTION_DIR = "src/casadi_functions"
# BATCH_SIZE = 1
# NUM_STEPS = 10

# device = "cuda"              # <-- must be cuda
# dtype = torch.double

# fn_casadi_sim_step = casadi.Function.load(
#     os.path.join(CUSADI_FUNCTION_DIR, "fn_sim_step.casadi")
# )
# fn_cusadi_sim_step = CusadiFunction(fn_casadi_sim_step, BATCH_SIZE)

# # State [theta, omega] on CUDA
# x = torch.tensor([[0.5, 0.0]], device=device, dtype=dtype)

# g  = 9.81 * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)
# l  = 1.0  * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)
# dt = 0.01 * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)

# print("Initial x:", x)

# for k in range(NUM_STEPS):
#     fn_cusadi_sim_step.evaluate([x, g, l, dt])
#     # torch.cuda.synchronize()
#     x = fn_cusadi_sim_step.outputs_sparse[0].clone()
#     print(f"step {k+1}: theta={x[0,0].item(): .6f}, omega={x[0,1].item(): .6f}")
