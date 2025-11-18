import os
import torch
import matplotlib.pyplot as plt
from casadi import *
from src import *  # CusadiFunction

# === 1. Settings ===
BATCH_SIZE = 4096          # number of Monte Carlo disturbance scenarios
N_PLOTS = 10               # number of quadrotors to log and plot
NUM_STEPS = 500
CUSADI_FUNCTION_DIR = "src/casadi_functions"

device = 'cuda'
dtype = torch.double

# === 2. Load CasADi and CusADi quadrotor functions ===
fn_casadi_quad_step = casadi.Function.load(
    os.path.join(CUSADI_FUNCTION_DIR, "fn_quad_sim_step.casadi")
)
fn_cusadi_quad_step = CusadiFunction(fn_casadi_quad_step, BATCH_SIZE)

print("Loaded CasADi function:", fn_casadi_quad_step)
print("CusADi quadrotor function loaded successfully!")
print(f"Running multi-step simulation for {NUM_STEPS} steps...")

# === 3. Batched initial conditions and parameters ===
# State: [p_x, p_z, theta, v_x, v_z, omega]
x0 = torch.zeros((BATCH_SIZE, 6), device=device, dtype=dtype)
x0[:, 1] = 1.0   # all start at p_z = 1m (same initial state for MC)

m_val = 1.0
I_val = 0.05
l_val = 0.2
g_val = 9.81

m = m_val * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)
I = I_val * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)
l = l_val * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)
g = g_val * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)

dt = 0.01 * torch.ones((BATCH_SIZE, 1), device=device, dtype=dtype)

# Control: hover thrust T1 = T2 = mg/2 (same for all scenarios here)
T_hover = m_val * g_val / 2.0
u = torch.zeros((BATCH_SIZE, 2), device=device, dtype=dtype)
u[:, 0] = T_hover
u[:, 1] = T_hover

# === 3b. Cost definition ===
# Reference state and control (hover)
x_ref = torch.zeros((1, 6), device=device, dtype=dtype)
x_ref[0, 1] = 1.0   # p_z = 1
u_ref = torch.tensor([[T_hover, T_hover]], device=device, dtype=dtype)

# Weights for quadratic cost
Q = torch.tensor([5.0, 5.0, 1.0, 0.1, 0.1, 0.1], device=device, dtype=dtype)
R = torch.tensor([0.01, 0.01], device=device, dtype=dtype)

# Cumulative cost per scenario (B,)
cum_cost = torch.zeros(BATCH_SIZE, device=device, dtype=dtype)

# === 4. Multi-step parallel simulation ===
x = x0.clone()
px_log = torch.zeros((NUM_STEPS, N_PLOTS), device=device, dtype=dtype)
pz_log = torch.zeros((NUM_STEPS, N_PLOTS), device=device, dtype=dtype)

wind_std = 0.2

for step in range(NUM_STEPS):
    # Resample disturbance each step for Monte Carlo
    w = wind_std * torch.randn((BATCH_SIZE, 2), device=device, dtype=dtype)

    # One-step dynamics
    fn_cusadi_quad_step.evaluate([x, u, w, m, I, l, g, dt])
    x = fn_cusadi_quad_step.outputs_sparse[0].clone()

    # --- Stage cost for each scenario ---
    x_err = x - x_ref      # broadcasts (B,6) - (1,6) -> (B,6)
    u_err = u - u_ref      # (B,2) - (1,2) -> (B,2)

    stage_cost = (Q * x_err**2).sum(dim=1) + (R * u_err**2).sum(dim=1)  # (B,)
    cum_cost += stage_cost

    # Log first N_PLOTS trajectories for visualization
    px_log[step] = x[:N_PLOTS, 0]
    pz_log[step] = x[:N_PLOTS, 1]

torch.cuda.synchronize()
print(f"✅ Quadrotor simulation complete ({NUM_STEPS} steps × {BATCH_SIZE} scenarios).")

# === 5. Monte Carlo statistics ===
J = cum_cost   # rename for clarity

J_mean = J.mean()
J_var = J.var(unbiased=False)

alpha = 0.9  # CVaR level
J_sorted, _ = torch.sort(J)
idx = int(alpha * BATCH_SIZE)
idx = min(max(idx, 0), BATCH_SIZE - 1)
VaR_alpha = J_sorted[idx]                 # empirical VaR_α
CVaR_alpha = J_sorted[idx:].mean()        # empirical CVaR_α (average of worst (1-α) tail)

print(f"Expected cost E[J]: {J_mean.item():.4f}")
print(f"Variance Var[J]:    {J_var.item():.4f}")
print(f"VaR_{alpha:.2f}:          {VaR_alpha.item():.4f}")
print(f"CVaR_{alpha:.2f}:         {CVaR_alpha.item():.4f}")

# Example combined risk-aware objective
lambda_var = 0.1
lambda_cvar = 0.5
risk_objective = J_mean + lambda_var * J_var + lambda_cvar * CVaR_alpha
print(f"Risk-aware objective: {risk_objective.item():.4f}")

# === 6. Plot trajectories for first N_PLOTS ===
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
