import os
import casadi as ca
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from src import CusadiFunction

print("\n=== Loading CartPole CasADi Model ===")

fn = ca.Function.load("src/casadi_functions/fn_cartpole_sim_step.casadi")

NX = 4
NU = 1

# physical parameters
m_c  = 1.0     # cart mass
m_p  = 0.3     # pole mass
l    = 0.5     # half-length
g    = 9.81
dt   = 0.01

# ============================================================
# 1 — Trim point for inverted equilibrium
# ============================================================

x_trim = np.array([0, 0, 0, 0], float)   # (p, p_dot, theta, theta_dot)
u_trim = np.array([0.0], float)          # zero force at upright equilibrium
w_zero = np.zeros(2)

print("\n=== CartPole Upright Trim ===")
print("x_trim:", x_trim)
print("u_trim:", u_trim)

# ============================================================
# 2 — Linearize Discrete Dynamics
# ============================================================

print("\n=== Linearizing System ===")

x_sym = ca.MX.sym("x", NX)
u_sym = ca.MX.sym("u", NU)
w_sym = ca.MX.sym("w", 2)

x_next = fn(x_sym, u_sym, w_sym, m_c, m_p, l, g, dt)
A_sym = ca.jacobian(x_next, x_sym)
B_sym = ca.jacobian(x_next, u_sym)

A_fn = ca.Function("A_fn", [x_sym, u_sym, w_sym], [A_sym])
B_fn = ca.Function("B_fn", [x_sym, u_sym, w_sym], [B_sym])

A = np.array(A_fn(x_trim, u_trim, w_zero))
B = np.array(B_fn(x_trim, u_trim, w_zero))

print("A:\n", A)
print("B:\n", B)

# ============================================================
# 3 — Infinite-Horizon Discrete LQR
# ============================================================

print("\n=== Solving DARE for LQR ===")

Q = np.diag([10, 1, 40, 5])     # strong on theta and theta_dot
R = np.diag([0.1])

P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

print("\n=== LQR Gain K ===")
print(K)

np.savez("cartpole_lqr_baseline.npz",
         A=A, B=B, K=K,
         x_trim=x_trim, u_trim=u_trim)

print("Saved cartpole_lqr_baseline.npz")

# ============================================================
# 4 — GPU Rollout Settings
# ============================================================

print("\n=== GPU Simulation ===")

BATCH = 4096
PLOTS = 10
STEPS = 500

device = "cuda"
dtype = torch.double

fn_casadi = ca.Function.load("src/casadi_functions/fn_cartpole_sim_step.casadi")
fn_gpu = CusadiFunction(fn_casadi, BATCH)

x_ref = torch.tensor(x_trim, dtype=dtype, device=device)
u_ref = torch.tensor(u_trim, dtype=dtype, device=device)
K_torch = torch.tensor(K, dtype=dtype, device=device)

# initial batch of states
x = torch.zeros((BATCH, NX), dtype=dtype, device=device)
x[:, 2] = 0.3    
x[:, 0] = 0.5

# logs
p_log = torch.zeros((STEPS, PLOTS), device=device)
theta_log = torch.zeros((STEPS, PLOTS), device=device)

# chance constraints
theta_max = 0.35
track_limit = 2.4

theta_violation_curve = torch.zeros((STEPS,), device=device)
track_violation_curve = torch.zeros((STEPS,), device=device)

# noise for evaluation
WIND_STD = 0.5

# parameters (batched)
m_c_t  = torch.full((BATCH,1), m_c, dtype=dtype, device=device)
m_p_t  = torch.full((BATCH,1), m_p, dtype=dtype, device=device)
l_t    = torch.full((BATCH,1), l,   dtype=dtype, device=device)
g_t    = torch.full((BATCH,1), g,   dtype=dtype, device=device)
dt_t   = torch.full((BATCH,1), dt,  dtype=dtype, device=device)

params = [m_c_t, m_p_t, l_t, g_t, dt_t]

print("Running rollout...")

for t in range(STEPS):

    # LQR control
    u = u_ref - (K_torch @ (x - x_ref).T).T

    # input limits
    u = torch.clamp(u, -3.0, 3.0)

    # disturbances
    w = WIND_STD * torch.randn((BATCH, 2), dtype=dtype, device=device)

    # apply dynamics
    fn_gpu.evaluate([x, u, w, *params])
    x = fn_gpu.outputs_sparse[0].clone()

    # log trajectories for first few
    p_log[t]     = x[:PLOTS, 0]
    theta_log[t] = x[:PLOTS, 2]

    # chance constraints
    theta_violation_curve[t] = (torch.abs(x[:,2]) > theta_max).double().mean()
    track_violation_curve[t] = (torch.abs(x[:,0]) > track_limit).double().mean()

torch.cuda.synchronize()
print("Simulation complete!")

# ============================================================
# 5 — Summary
# ============================================================

print("\n================ Chance Constraints Summary ================")
print("Final pole-angle violation prob =", theta_violation_curve[-1].item())
print("Final track violation prob      =", track_violation_curve[-1].item())
print("============================================================")

# ============================================================
# 6 — Plot trajectories
# ============================================================

p_np = p_log.cpu().numpy()
theta_np = theta_log.cpu().numpy()

fig, ax = plt.subplots(1, 2, figsize=(13,5))

for i in range(PLOTS):
    ax[0].plot(p_np[:,i])
    ax[1].plot(theta_np[:,i])

ax[0].set_title("Cart Position p — LQR")
ax[0].set_xlabel("Time Step")
ax[1].set_ylabel("p")

ax[1].set_title("Pole Angle θ — LQR")
ax[1].set_xlabel("Time Step")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/cartpole_lqr_trajs.png", dpi=150)
plt.show()
print("Saved trajectory plots.")

# ============================================================
# 7 — Chance constraint plots
# ============================================================

plt.figure(figsize=(10,5))
plt.plot(theta_violation_curve.cpu(), label="Pole angle violation prob")
plt.plot(track_violation_curve.cpu(), label="Track violation prob")
plt.axhline(0.05, color='red', linestyle='--', label="α = 0.05")

plt.title("Chance Constraint Violations — CartPole LQR")
plt.xlabel("Time Step")
plt.ylabel("Probability")
plt.grid(True)
plt.legend()

plt.savefig("results/cartpole_lqr_chance.png", dpi=150)
plt.show()
print("Saved chance-constraint plots.")
