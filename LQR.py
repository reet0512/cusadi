import casadi as ca
import numpy as np
from scipy.linalg import solve_discrete_are
import torch
import matplotlib.pyplot as plt
from src import CusadiFunction

# ================================================================
# SECTION 1 — Load CasADi quadrotor step function
# ================================================================
fn = ca.Function.load("src/casadi_functions/fn_quad_sim_step.casadi")

nx = 6
nu = 2

# Physical parameters
m  = 1.0
I  = 0.05
l  = 0.2
g  = 9.81
dt = 0.01

# ================================================================
# SECTION 2 — Trim hover equilibrium
# ================================================================
T_hover = m * g
T1_trim = T_hover / 2
T2_trim = T_hover / 2

x_trim = np.array([0., 1., 0., 0., 0., 0.])
u_trim = np.array([T1_trim, T2_trim])

print("=== Hover Trim ===")
print("x_trim:", x_trim)
print("u_trim:", u_trim)

# ================================================================
# SECTION 3 — Linearization of discrete dynamics
# ================================================================
x_sym = ca.MX.sym("x", nx)
u_sym = ca.MX.sym("u", nu)
w_sym = ca.MX.sym("w", 2)

x_next = fn(x_sym, u_sym, w_sym, m, I, l, g, dt)

A_sym = ca.jacobian(x_next, x_sym)
B_sym = ca.jacobian(x_next, u_sym)

A_fn = ca.Function("A_fn", [x_sym, u_sym, w_sym], [A_sym])
B_fn = ca.Function("B_fn", [x_sym, u_sym, w_sym], [B_sym])

A = np.array(A_fn(x_trim, u_trim, np.zeros(2)))
B = np.array(B_fn(x_trim, u_trim, np.zeros(2)))

# A = np.array(A_fn(x_trim, u_trim, np.zeros(2), m, I, l, g, dt))
# B = np.array(B_fn(x_trim, u_trim, np.zeros(2), m, I, l, g, dt))


print("\n=== Linearized System ===")
print("A:\n", A)
print("B:\n", B)

# ================================================================
# SECTION 4 — LQR Gain
#   We solve for Δu = -K (x - x_trim)
#   Final control is: u = u_trim + Δu
# ================================================================
Q = np.diag([40, 40, 10, 2, 2, 1])
# Q = np.diag([80, 80, 40, 4, 4, 2])
R = np.diag([0.4, 0.4])

P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

print("\n=== LQR Gain ===")
print(K)

np.savez("lqr_baseline.npz", A=A, B=B, K=K, x_trim=x_trim, u_trim=u_trim)
print("\nSaved lqr_baseline.npz")

# ================================================================
# SECTION 5 — GPU Simulation
# ================================================================
print("\n=== GPU Simulation ===")

BATCH = 4096
PLOTS = 10
STEPS = 500

device = "cuda"
dtype = torch.double

fn_casadi = ca.Function.load("src/casadi_functions/fn_quad_sim_step.casadi")
fn_gpu = CusadiFunction(fn_casadi, BATCH)

# Convert numpy refs → torch refs
x_ref = torch.tensor(x_trim, device=device, dtype=dtype)
u_ref = torch.tensor(u_trim, device=device, dtype=dtype)
K_torch = torch.tensor(K, device=device, dtype=dtype)

# Initial state batch
x = torch.zeros((BATCH, nx), dtype=dtype, device=device)
x[:, 1] = 1.0  # altitude
x[:, 2] = 0.0  # level

px_log = torch.zeros((STEPS, PLOTS), device=device)
pz_log = torch.zeros((STEPS, PLOTS), device=device)

wind_std = 0.2

# ================================================================
# LQR rollout — CORRECTED
# ================================================================
for t in range(STEPS):

    # LQR computes Δu
    du = -(K_torch @ (x - x_ref).T).T

    # IMPORTANT: scale control (quadrotor is very weakly actuated)
    du = 5.0 * du

    # Absolute thrust
    u = u_ref + du

    # Safe clamp
    u = torch.clamp(u, 0.0, 20.0)

    # Disturbance
    w = wind_std * torch.randn((BATCH, 2), device=device, dtype=dtype)

    # Parameters
    params = [
        torch.full((BATCH, 1), m, dtype=dtype, device=device),
        torch.full((BATCH, 1), I, dtype=dtype, device=device),
        torch.full((BATCH, 1), l, dtype=dtype, device=device),
        torch.full((BATCH, 1), g, dtype=dtype, device=device),
        torch.full((BATCH, 1), dt, dtype=dtype, device=device),
    ]

    fn_gpu.evaluate([x, u, w, *params])
    x = fn_gpu.outputs_sparse[0].clone()

    # if t < 20:
    #     print(t, 
    #     u[:,0].mean().item(), u[:,1].mean().item(),
    #     x[:,1].mean().item(),  # mean p_z
    #     x[:,4].mean().item())  # mean v_z


    px_log[t] = x[:PLOTS, 0]
    pz_log[t] = x[:PLOTS, 1]

torch.cuda.synchronize()
print("Simulation complete!")

# ================================================================
# SECTION 6 — Plot Results
# ================================================================
px = px_log.cpu().numpy()
pz = pz_log.cpu().numpy()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

for i in range(PLOTS):
    ax[0].plot(px[:, i])
    ax[1].plot(pz[:, i])

ax[0].set_title("Horizontal Position (px) — LQR Baseline")
ax[0].set_xlabel("Time Step")
ax[0].set_ylabel("px (m)")
ax[0].grid(True)

ax[1].set_title("Vertical Position (pz) — LQR Baseline")
ax[1].set_xlabel("Time Step")
ax[1].set_ylabel("pz (m)")
ax[1].grid(True)

plt.tight_layout()
plt.savefig("results/lqr_baseline_trajs.png", dpi=150)
plt.show()
print("Figure saved to results/lqr_baseline_trajs.png")