import os
import casadi as ca
import numpy as np
from scipy.linalg import solve_discrete_are
import torch
import matplotlib.pyplot as plt
from src import CusadiFunction

# =====================================================================
# SECTION 1 — Load 3D Model and Physical Parameters
# =====================================================================
print("\n=== Loading 3D Quadrotor CasADi Model ===")

CUSADI_FUNCTION_DIR = "src/casadi_functions"
fn = ca.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "fn_quad_sim_step_3d.casadi"))

nx = 12   # [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
nu = 4    # [T1, T2, T3, T4]

# physical constants
m  = 1.0
Jx = 0.02
Jy = 0.02
Jz = 0.04
l  = 0.2
k_m = 0.01
g  = 9.81
dt = 0.01

# =====================================================================
# SECTION 2 — Hover Trim Point
# =====================================================================
# Hover at origin, 1m altitude, zero velocity, zero angles, zero rates
x_trim = np.array([
    0.0, 0.0, 1.0,   # x, y, z
    0.0, 0.0, 0.0,   # vx, vy, vz
    0.0, 0.0, 0.0,   # phi, theta, psi
    0.0, 0.0, 0.0    # p, q, r
], float)

T_hover = m * g
T_each  = T_hover / 4.0
u_trim = np.array([T_each, T_each, T_each, T_each], float)

w_zero = np.zeros(3)

print("\n=== Hover Trim (3D) ===")
print("x_trim:", x_trim)
print("u_trim:", u_trim)

# =====================================================================
# SECTION 3 — Linearize Dynamics
# =====================================================================
print("\n=== Linearizing 3D System at Hover ===")

x_sym = ca.MX.sym("x", nx)
u_sym = ca.MX.sym("u", nu)
w_sym = ca.MX.sym("w", 3)

x_next = fn(
    x_sym,
    u_sym,
    w_sym,
    m, Jx, Jy, Jz,
    l, k_m, g,
    dt
)

A_sym = ca.jacobian(x_next, x_sym)
B_sym = ca.jacobian(x_next, u_sym)

A_fn = ca.Function("A_fn", [x_sym, u_sym, w_sym], [A_sym])
B_fn = ca.Function("B_fn", [x_sym, u_sym, w_sym], [B_sym])

A = np.array(A_fn(x_trim, u_trim, w_zero))
B = np.array(B_fn(x_trim, u_trim, w_zero))

print("A shape:", A.shape)
print("B shape:", B.shape)

# =====================================================================
# SECTION 4 — Infinite-Horizon LQR
# =====================================================================
print("\n=== Solving Discrete Algebraic Riccati Equation (DARE) ===")

# Q penalizes [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
Q = np.diag([
    10.0, 10.0, 40.0,  # position x,y,z
    2.0,  2.0,  5.0,   # velocity
    20.0, 20.0,  5.0,  # roll, pitch, yaw
    1.0,  1.0,  1.0    # body rates p,q,r
])

# R penalizes 4 thrusts
R = np.diag([0.5, 0.5, 0.5, 0.5])

P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

print("\n=== LQR Gain (3D) ===")
print(K)

np.savez("lqr_baseline_3d.npz", A=A, B=B, K=K, x_hover=x_trim, u_hover=u_trim)
print("Saved lqr_baseline_3d.npz")

# =====================================================================
# SECTION 5 — GPU Rollout Settings
# =====================================================================
print("\n=== GPU Simulation (3D) ===")

BATCH = 4096
PLOTS = 10
STEPS = 500

device = "cuda"
dtype = torch.double

fn_casadi = ca.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "fn_quad_sim_step_3d.casadi"))
fn_gpu = CusadiFunction(fn_casadi, BATCH)

x_ref = torch.tensor(x_trim, dtype=dtype, device=device)
u_ref = torch.tensor(u_trim, dtype=dtype, device=device)
K_torch = torch.tensor(K, dtype=dtype, device=device)

# initial state distribution
x = torch.zeros((BATCH, nx), dtype=dtype, device=device)
x[:, 2] = 1.0    # starting altitude z
# small random perturbation around origin
x[:, 0] = 0.1 * torch.randn(BATCH, dtype=dtype, device=device)  # x
x[:, 1] = 0.1 * torch.randn(BATCH, dtype=dtype, device=device)  # y

# logs (for first PLOTS trajectories)
px_log = torch.zeros((STEPS, PLOTS), device=device)
py_log = torch.zeros((STEPS, PLOTS), device=device)
pz_log = torch.zeros((STEPS, PLOTS), device=device)
phi_log = torch.zeros((STEPS, PLOTS), device=device)
theta_log = torch.zeros((STEPS, PLOTS), device=device)

# chance constraint logs
tilt_violation_t = torch.zeros((STEPS,), device=device)
alt_violation_t  = torch.zeros((STEPS,), device=device)

# noise
wind_std = 0.2

# parameters (batched)
m_t   = torch.full((BATCH,1), m,   dtype=dtype, device=device)
Jx_t  = torch.full((BATCH,1), Jx,  dtype=dtype, device=device)
Jy_t  = torch.full((BATCH,1), Jy,  dtype=dtype, device=device)
Jz_t  = torch.full((BATCH,1), Jz,  dtype=dtype, device=device)
l_t   = torch.full((BATCH,1), l,   dtype=dtype, device=device)
km_t  = torch.full((BATCH,1), k_m, dtype=dtype, device=device)
g_t   = torch.full((BATCH,1), g,   dtype=dtype, device=device)
dt_t  = torch.full((BATCH,1), dt,  dtype=dtype, device=device)

# chance constraint safety limits
tilt_max = 0.35  # max sqrt(phi^2 + theta^2)
z_min    = 1.0   # minimum safe altitude


# =====================================================================
# SECTION 6 — Rollout
# =====================================================================
print("Running 3D LQR rollout...")

for t in range(STEPS):

    # LQR control in deviation coordinates
    # u = u_ref - K (x - x_ref)
    u = u_ref - (K_torch @ (x - x_ref).T).T

    # thrust saturation
    u = torch.clamp(u, 0.0, 20.0)

    # disturbances (world-frame wind)
    w = wind_std * torch.randn((BATCH, 3), dtype=dtype, device=device)

    # step dynamics
    fn_gpu.evaluate([x, u, w, m_t, Jx_t, Jy_t, Jz_t, l_t, km_t, g_t, dt_t])
    x = fn_gpu.outputs_sparse[0].clone()

    # logging first few trajectories
    px_log[t]    = x[:PLOTS, 0]  # x
    py_log[t]    = x[:PLOTS, 1]  # y
    pz_log[t]    = x[:PLOTS, 2]  # z
    phi_log[t]   = x[:PLOTS, 6]  # roll
    theta_log[t] = x[:PLOTS, 7]  # pitch

    # chance constraint measurements
    # tilt_angle = sqrt(phi^2 + theta^2)
    tilt_angle = torch.sqrt(x[:,6]**2 + x[:,7]**2)
    tilt_violation_t[t] = (tilt_angle > tilt_max).double().mean()
    alt_violation_t[t]  = (x[:,2] < z_min).double().mean()

torch.cuda.synchronize()
print("3D LQR Simulation complete!")


# =====================================================================
# SECTION 7 — Chance Constraint Summary
# =====================================================================
tilt_prob = tilt_violation_t[-1].item()
alt_prob  = alt_violation_t[-1].item()

print("\n================ 3D Chance Constraint Summary ================")
print(f"Tilt violation probability   = {tilt_prob:.4f}")
print(f"Altitude violation prob      = {alt_prob:.4f}")
print("==============================================================")

# =====================================================================
# SECTION 8 — Plot Trajectories (x, y, z)
# =====================================================================
px = px_log.cpu().numpy()
py = py_log.cpu().numpy()
pz = pz_log.cpu().numpy()

fig, ax = plt.subplots(1,3, figsize=(15,4))

for i in range(PLOTS):
    ax[0].plot(px[:,i])
    ax[1].plot(py[:,i])
    ax[2].plot(pz[:,i])

ax[0].set_title("x Position — 3D LQR")
ax[0].set_xlabel("Timestep")
ax[0].set_ylabel("x (m)")

ax[1].set_title("y Position — 3D LQR")
ax[1].set_xlabel("Timestep")
ax[1].set_ylabel("y (m)")

ax[2].set_title("z Position — 3D LQR")
ax[2].set_xlabel("Timestep")
ax[2].set_ylabel("z (m)")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/lqr_3d_baseline_trajs.png", dpi=150)
plt.show()
print("Figure saved to results/lqr_3d_baseline_trajs.png")

# =====================================================================
# SECTION 9 — Plot Chance Violations
# =====================================================================
plt.figure(figsize=(10,4))
plt.plot(tilt_violation_t.cpu(), label="Tilt violation probability")
plt.plot(alt_violation_t.cpu(),  label="Altitude violation probability")
plt.axhline(0.05, color='red', linestyle="--", label="α = 0.05")

plt.title("Chance Constraint Violations Over Time — 3D LQR Baseline")
plt.xlabel("Time Step")
plt.ylabel("Probability")
plt.grid(True)
plt.legend()

plt.savefig("results/lqr_3d_chance_constraint_plot.png", dpi=150)
plt.show()
print("Figure saved to results/lqr_3d_chance_constraint_plot.png")
