import os
import casadi as ca
import numpy as np
from scipy.linalg import solve_discrete_are
import torch
import matplotlib.pyplot as plt
from src import CusadiFunction


# =====================================================================
# SECTION 1 — Load Model and Physical Parameters
# =====================================================================
print("\n=== Loading Quadrotor CasADi Model ===")

fn = ca.Function.load("src/casadi_functions/fn_quad_sim_step.casadi")

nx = 6
nu = 2

# physical constants
m  = 1.0
I  = 0.05
l  = 0.2
g  = 9.81
dt = 0.01

# =====================================================================
# SECTION 2 — Hover Trim Point
# =====================================================================
x_trim = np.array([0, 1, 0, 0, 0, 0], float)   # (px, pz, theta, vx, vz, omega)
u_trim = np.array([m * g / 2, m * g / 2], float)
w_zero = np.zeros(2)

print("\n=== Hover Trim ===")
print("x_trim:", x_trim)
print("u_trim:", u_trim)

# =====================================================================
# SECTION 3 — Linearize Dynamics
# =====================================================================
print("\n=== Linearizing System at Hover ===")

x_sym = ca.MX.sym("x", nx)
u_sym = ca.MX.sym("u", nu)
w_sym = ca.MX.sym("w", 2)

x_next = fn(x_sym, u_sym, w_sym, m, I, l, g, dt)
A_sym = ca.jacobian(x_next, x_sym)
B_sym = ca.jacobian(x_next, u_sym)

A_fn = ca.Function("A_fn", [x_sym, u_sym, w_sym], [A_sym])
B_fn = ca.Function("B_fn", [x_sym, u_sym, w_sym], [B_sym])

A = np.array(A_fn(x_trim, u_trim, w_zero))
B = np.array(B_fn(x_trim, u_trim, w_zero))

print("A:\n", A)
print("B:\n", B)

# =====================================================================
# SECTION 4 — Infinite-Horizon LQR
# =====================================================================
print("\n=== Solving Discrete Algebraic Riccati Equation (DARE) ===")

Q = np.diag([40, 40, 10, 2, 2, 1])
R = np.diag([0.4, 0.4])

P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

print("\n=== LQR Gain ===")
print(K)

np.savez("lqr_baseline.npz", A=A, B=B, K=K, x_hover=x_trim, u_hover=u_trim)
print("Saved lqr_baseline.npz")

# =====================================================================
# SECTION 5 — GPU Rollout Settings
# =====================================================================
print("\n=== GPU Simulation ===")

BATCH = 4096
PLOTS = 10
STEPS = 500

device = "cuda"
dtype = torch.double

fn_casadi = ca.Function.load("src/casadi_functions/fn_quad_sim_step.casadi")
fn_gpu = CusadiFunction(fn_casadi, BATCH)

x_ref = torch.tensor(x_trim, dtype=dtype, device=device)
u_ref = torch.tensor(u_trim, dtype=dtype, device=device)
K_torch = torch.tensor(K, dtype=dtype, device=device)

# initial state distribution
x = torch.zeros((BATCH, nx), dtype=dtype, device=device)
x[:, 1] = 1.0   # starting altitude
x[:, 2] = 0.0   # starting orientation

# logs
px_log = torch.zeros((STEPS, PLOTS), device=device)
pz_log = torch.zeros((STEPS, PLOTS), device=device)

# chance constraint logs
tilt_violation_t = torch.zeros((STEPS,), device=device)
alt_violation_t  = torch.zeros((STEPS,), device=device)

# noise
wind_std = 0.2

# parameters (batched)
m_t  = torch.full((BATCH,1), m, dtype=dtype, device=device)
I_t  = torch.full((BATCH,1), I, dtype=dtype, device=device)
l_t  = torch.full((BATCH,1), l, dtype=dtype, device=device)
g_t  = torch.full((BATCH,1), g, dtype=dtype, device=device)
dt_t = torch.full((BATCH,1), dt, dtype=dtype, device=device)

# chance constraint safety limits
theta_max = 0.35
z_min     = 0.3


# =====================================================================
# SECTION 6 — Rollout
# =====================================================================
print("Running rollout...")

for t in range(STEPS):

    # LQR control
    u = u_ref - (K_torch @ (x - x_ref).T).T

    # thrust saturation
    u = torch.clamp(u, 0.0, 15.0)

    # disturbances
    w = wind_std * torch.randn((BATCH, 2), dtype=dtype, device=device)

    # step dynamics
    fn_gpu.evaluate([x, u, w, m_t, I_t, l_t, g_t, dt_t])
    x = fn_gpu.outputs_sparse[0].clone()

    # logging first few trajectories
    px_log[t] = x[:PLOTS, 0]
    pz_log[t] = x[:PLOTS, 1]

    # chance constraint measurements
    tilt_violation_t[t] = (torch.abs(x[:,2]) > theta_max).double().mean()
    alt_violation_t[t]  = (x[:,1] < z_min).double().mean()

torch.cuda.synchronize()
print("Simulation complete!")


# =====================================================================
# SECTION 7 — Chance Constraint Summary
# =====================================================================
tilt_prob = tilt_violation_t[-1].item()
alt_prob  = alt_violation_t[-1].item()

print("\n================ Chance Constraint Summary ================")
print(f"Tilt violation probability   = {tilt_prob:.4f}")
print(f"Altitude violation prob      = {alt_prob:.4f}")
print("===========================================================")


# =====================================================================
# SECTION 8 — Plot Trajectories
# =====================================================================
px = px_log.cpu().numpy()
pz = pz_log.cpu().numpy()

fig, ax = plt.subplots(1,2, figsize=(13,5))

for i in range(PLOTS):
    ax[0].plot(px[:,i])
    ax[1].plot(pz[:,i])

ax[0].set_title("Horizontal Position px — LQR")
ax[0].set_xlabel("Timestep")
ax[0].set_ylabel("px")

ax[1].set_title("Vertical Position pz — LQR")
ax[1].set_xlabel("Timestep")
ax[1].set_ylabel("pz")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/lqr_baseline_trajs.png", dpi=150)
plt.show()
print("Figure saved to results/lqr_baseline_trajs.png")

# =====================================================================
# SECTION 9 — Plot Chance Violations
# =====================================================================
plt.figure(figsize=(10,4))
plt.plot(tilt_violation_t.cpu(), label="Tilt violation probability")
plt.plot(alt_violation_t.cpu(), label="Altitude violation probability")
plt.axhline(0.05, color='red', linestyle="--", label="α = 0.05")

plt.title("Chance Constraint Violations Over Time — LQR Baseline")
plt.xlabel("Time Step")
plt.ylabel("Probability")
plt.grid(True)
plt.legend()

plt.savefig("results/lqr_chance_constraint_plot.png", dpi=150)
plt.show()
print("Figure saved to results/lqr_chance_constraint_plot.png")