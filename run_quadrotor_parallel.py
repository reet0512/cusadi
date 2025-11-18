import os
import torch
import matplotlib.pyplot as plt
from casadi import *
from src import *  # assumes CusadiFunction etc. are here

# === 1. Settings ===
BATCH_SIZE = 4096

N_PLOTS = 10              # number of quadrotors to log and plot
NUM_STEPS = 500
CUSADI_FUNCTION_DIR = "src/casadi_functions"

# === 2. Load CasADi and CusADi quadrotor functions ===
fn_casadi_quad_step = casadi.Function.load(
    os.path.join(CUSADI_FUNCTION_DIR, "fn_quad_sim_step.casadi")
)

fn_cusadi_quad_step = CusadiFunction(fn_casadi_quad_step, BATCH_SIZE)

print("Loaded CasADi function:", fn_casadi_quad_step)
print("CusADi quadrotor function loaded successfully!")
print(f"Running multi-step simulation for {NUM_STEPS} steps...")

# === 3. Create batched initial conditions and parameters (on GPU) ===
device = 'cuda'
dtype = torch.double

# State: [p_x, p_z, theta, v_x, v_z, omega]
x0 = torch.zeros((BATCH_SIZE, 6), device=device, dtype=dtype)
# small random perturbation from hover
x0[:, 0] = torch.linspace(-0.5, 0.5, BATCH_SIZE, device=device, dtype=dtype)  # p_x
x0[:, 1] = 1.0  # start 1m above ground
# others left at zero

# Physical parameters (batched to match CusADi style)
m_val = 1.0
I_val = 0.05
l_val = 0.2
g_val = 9.81

m = m_val * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)
I = I_val * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)
l = l_val * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)
g = g_val * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)

# Time step (can be constant or vary per batch)
dt = 0.01 * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)

# Control: hover thrust T1 = T2 = mg/2
T_hover = m_val * g_val / 2.0
u = torch.zeros((BATCH_SIZE, 2), device=device, dtype=dtype)
u[:, 0] = T_hover
u[:, 1] = T_hover

# Disturbance: random wind [w_x, w_z]
# For now: i.i.d. Gaussian; in MPC you would resample per step
wind_std = 0.2
w = wind_std * torch.randn((BATCH_SIZE, 2), device=device, dtype=dtype)

# === 4. Multi-step parallel simulation ===
x = x0.clone()
px_log = torch.zeros((NUM_STEPS, N_PLOTS), device=device, dtype=dtype)
pz_log = torch.zeros((NUM_STEPS, N_PLOTS), device=device, dtype=dtype)

print(fn_casadi_quad_step)

for step in range(NUM_STEPS):
    # Option 1: fixed disturbance per env
    # Option 2 (more stochastic): resample w each step:
    # w = wind_std * torch.randn((BATCH_SIZE, 2), device=device, dtype=dtype)

    fn_cusadi_quad_step.evaluate([x, u, w, m, I, l, g, dt])
    x = fn_cusadi_quad_step.outputs_sparse[0].clone()

    # log first N_PLOTS trajectories
    px_log[step] = x[:N_PLOTS, 0]  # p_x
    pz_log[step] = x[:N_PLOTS, 1]  # p_z

torch.cuda.synchronize()
print(f"✅ Quadrotor simulation complete ({NUM_STEPS} steps × {BATCH_SIZE} quads).")

# === 5. Plot trajectories for first N_PLOTS quadrotors ===
px_cpu = px_log.cpu().numpy()
pz_cpu = pz_log.cpu().numpy()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

for i in range(N_PLOTS):
    ax[0].plot(px_cpu[:, i], label=f'Quad {i}')
    ax[1].plot(pz_cpu[:, i], label=f'Quad {i}')

ax[0].set_xlabel("Time step")
ax[0].set_ylabel("p_x (m)")
ax[0].set_title("Horizontal position trajectories")

ax[1].set_xlabel("Time step")
ax[1].set_ylabel("p_z (m)")
ax[1].set_title("Vertical position trajectories")

for a in ax:
    a.grid(True)
    a.legend(fontsize=7)

fig.tight_layout()
os.makedirs('results', exist_ok=True)
fig_path = os.path.join('results', f'quadrotor_trajectories_first_{N_PLOTS}.png')
fig.savefig(fig_path, dpi=150)
print(f"✅ Figure saved to: {fig_path}")
plt.show()
