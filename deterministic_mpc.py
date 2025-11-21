import casadi as ca
import numpy as np
import torch
import matplotlib.pyplot as plt
from src import CusadiFunction


# ============================================================
# 1. Load quadrotor model (CasADi)
# ============================================================
fn = ca.Function.load("src/casadi_functions/fn_quad_sim_step.casadi")

nx = 6     # state dimension
nu = 2     # control dimension


# ============================================================
# 2. MPC horizon and weights
# ============================================================
N = 20             # horizon length
dt = 0.01

Q = np.diag([10, 10, 2, 0.5, 0.5, 0.5])
R = np.diag([0.1, 0.1])
Qf = np.diag([20, 20, 5, 1, 1, 1])

x_ref = np.array([0, 1, 0, 0, 0, 0])         # hover
u_ref = np.array([4.905, 4.905])             # m*g/2 each


# ============================================================
# 3. Build CasADi MPC Optimization Problem
# ============================================================
opti = ca.Opti()

X = opti.variable(nx, N+1)   # state trajectory
U = opti.variable(nu, N)     # control trajectory

x0_param = opti.parameter(nx)   # initial state parameter


# Dynamics constraints
def step(x, u):
    return fn(x, u, np.zeros(2), 1.0, 0.05, 0.2, 9.81, dt)

opti.subject_to(X[:, 0] == x0_param)

for k in range(N):
    opti.subject_to(X[:, k+1] == step(X[:, k], U[:, k]))


# Objective function
cost = 0
for k in range(N):
    dx = X[:, k] - x_ref
    du = U[:, k] - u_ref
    cost += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([du.T, R, du])

dxN = X[:, N] - x_ref
cost += ca.mtimes([dxN.T, Qf, dxN])

opti.minimize(cost)

# (Optional) Bound controls
# opti.subject_to(U >= 0)
# opti.subject_to(U <= 12)
# Control bounds (elementwise)
opti.subject_to(opti.bounded(0, U, 12))



# Solver
opti.solver("ipopt")


# ============================================================
# 4. Simulation settings (GPU rollout for real dynamics)
# ============================================================
STEPS = 300
BATCH_SIZE = 4096   # for GPU rollout of next state

fn_casadi = fn
fn_cusadi = CusadiFunction(fn_casadi, BATCH_SIZE)

device = "cuda"
dtype = torch.double

x_sim = torch.zeros((BATCH_SIZE, nx), device=device, dtype=dtype)
x_sim[:, 1] = 1.0   # start at p_z = 1m

px_log = []
pz_log = []


# ============================================================
# 5. Closed-loop MPC simulation
# ============================================================
for t in range(STEPS):

    # ---- Solve MPC for the FIRST quadrotor ----
    opti.set_value(x0_param, x_sim[0].cpu().numpy())

    try:
        sol = opti.solve()
    except:
        sol = opti.debug

    u0 = sol.value(U[:, 0])     # first control action

    # ---- Apply u0 to the BATCH of 4096 environments ----
    u_batch = torch.tensor(u0, dtype=torch.double, device=device).repeat(BATCH_SIZE, 1)
    w_batch = torch.zeros((BATCH_SIZE, 2), dtype=torch.double, device=device)

    m  = torch.ones((BATCH_SIZE,1), device=device)*1.0
    I  = torch.ones((BATCH_SIZE,1), device=device)*0.05
    l  = torch.ones((BATCH_SIZE,1), device=device)*0.2
    g  = torch.ones((BATCH_SIZE,1), device=device)*9.81
    dt_t = torch.ones((BATCH_SIZE,1), device=device)*dt

    fn_cusadi.evaluate([x_sim, u_batch, w_batch, m, I, l, g, dt_t])
    x_sim = fn_cusadi.outputs_sparse[0].clone()

    px_log.append(float(x_sim[0,0]))
    pz_log.append(float(x_sim[0,1]))


# ============================================================
# 6. Plot results
# ============================================================
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(px_log)
plt.title("px trajectory — Deterministic MPC")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(pz_log)
plt.title("pz trajectory — Deterministic MPC")
plt.grid(True)

plt.tight_layout()
plt.savefig("results/mpc_deterministic.png", dpi=150)
plt.show()

print("✓ Saved MPC plot to results/mpc_deterministic.png")
