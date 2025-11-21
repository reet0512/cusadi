import casadi as ca
import numpy as np
import torch
import matplotlib.pyplot as plt
from src import CusadiFunction


# ================================================================
# SECTION 1 — Load discrete CasADi quadrotor step
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

# parameter CasADi symbols
m_sym  = ca.SX(m)
I_sym  = ca.SX(I)
l_sym  = ca.SX(l)
g_sym  = ca.SX(g)
dt_sym = ca.SX(dt)


# ================================================================
# SECTION 2 — Hover equilibrium (same as LQR)
# ================================================================
T_hover = m * g
x_trim = np.array([0., 1., 0., 0., 0., 0.])
u_trim = np.array([T_hover/2, T_hover/2])

print("=== MPC Hover Trim ===")
print("x_trim:", x_trim)
print("u_trim:", u_trim)


# ================================================================
# SECTION 3 — MPC Problem Setup
# ================================================================
N = 30  # horizon length

# Decision variables
X = ca.SX.sym("X", nx, N+1)
U = ca.SX.sym("U", nu, N)

# Parameters: initial state
X0 = ca.SX.sym("X0", nx)

# Objective
Q = np.diag([40, 40, 10, 2, 2, 1])
R = np.diag([0.4, 0.4])

cost = 0

# Build MPC cost
for k in range(N):
    dx = X[:, k] - x_trim
    du = U[:, k] - u_trim
    cost += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([du.T, R, du])
    # small terminal cost
dxN = X[:, N] - x_trim
cost += ca.mtimes([dxN.T, Q, dxN])

# Constraints list
g_cons = []

# Initial condition constraint
g_cons.append(X[:,0] - X0)

# System dynamics constraints (SAME fn as LQR)
for k in range(N):
    x_k   = X[:, k]
    u_k   = U[:, k]
    w_k   = ca.DM.zeros(2)       # deterministic MPC → no wind

    x_next_k = fn(
        x_k,
        u_k,
        w_k,
        m_sym,
        I_sym,
        l_sym,
        g_sym,
        dt_sym
    )

    g_cons.append(X[:, k+1] - x_next_k)

# Control constraints (same as LQR)
u_min = np.array([0.0, 0.0])
u_max = np.array([20.0, 20.0])

lbX = -ca.inf * np.ones(X.shape)
ubX =  ca.inf * np.ones(X.shape)

lbU = np.tile(u_min.reshape(-1,1), (1,N))
ubU = np.tile(u_max.reshape(-1,1), (1,N))

# Flatten decision vars
opt_vars = ca.vertcat(ca.vec(X), ca.vec(U))
g_flat   = ca.vertcat(*g_cons)

lb_vars = []
ub_vars = []

lb_vars.append(lbX)
ub_vars.append(ubX)
lb_vars.append(lbU)
ub_vars.append(ubU)

lb_vars = np.concatenate([v.flatten() for v in lb_vars])
ub_vars = np.concatenate([v.flatten() for v in ub_vars])

lb_g = np.zeros(g_flat.shape)
ub_g = np.zeros(g_flat.shape)

# Build solver
nlp = {
    "x": opt_vars,
    "f": cost,
    "g": g_flat,
    "p": X0
}

solver = ca.nlpsol(
    "solver", "ipopt", nlp,
    {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 1000
    }
)

print("\n=== MPC Solver Built ===")


# ================================================================
# SECTION 4 — GPU Simulation loop using MPC
# ================================================================
BATCH = 4096
PLOTS = 10
STEPS = 500

device = "cuda"
dtype = torch.double

fn_gpu = CusadiFunction(fn, BATCH)

# Convert trims to torch
x_ref = torch.tensor(x_trim, device=device, dtype=dtype)
u_ref = torch.tensor(u_trim, device=device, dtype=dtype)

# Initial state batch
x = torch.zeros((BATCH, nx), device=device, dtype=dtype)
x[:, 1] = 1.0  # altitude
# x[:, 2] = 0.1  # slight tilt (for meaningful MPC correction)

px_log = torch.zeros((STEPS, PLOTS), device=device)
pz_log = torch.zeros((STEPS, PLOTS), device=device)

for t in range(STEPS):

    # Solve MPC for the first environment only (deterministic baseline)
    x0_np = x[0].cpu().numpy()

    # initial guess
    x_guess = np.tile(x0_np.reshape(-1,1), (1, N+1))
    u_guess = np.tile(u_trim.reshape(-1,1), (1, N))

    sol = solver(
        x0 = np.concatenate([x_guess.flatten(), u_guess.flatten()]),
        lbx = lb_vars,
        ubx = ub_vars,
        lbg = lb_g,
        ubg = ub_g,
        p = x0_np
    )

    opt = sol["x"].full().flatten()

    # Extract U0 MPC action
    X_flat = opt[: nx*(N+1)]
    U_flat = opt[nx*(N+1):]

    Umat = U_flat.reshape((nu, N))
    u_mpc = torch.tensor(Umat[:,0], device=device, dtype=dtype)

    # Broadcast MPC action across batch
    u = u_mpc.repeat(BATCH, 1)

    # Disturbance for fairness (same as LQR baseline)
    w = 0.2 * torch.randn((BATCH, 2), device=device, dtype=dtype)

    # Parameters
    params = [
        torch.full((BATCH,1), m, dtype=dtype, device=device),
        torch.full((BATCH,1), I, dtype=dtype, device=device),
        torch.full((BATCH,1), l, dtype=dtype, device=device),
        torch.full((BATCH,1), g, dtype=dtype, device=device),
        torch.full((BATCH,1), dt, dtype=dtype, device=device),
    ]

    fn_gpu.evaluate([x, u, w, *params])
    x = fn_gpu.outputs_sparse[0].clone()

    px_log[t] = x[:PLOTS, 0]
    pz_log[t] = x[:PLOTS, 1]

torch.cuda.synchronize()
print("Simulation complete!")


# ================================================================
# SECTION 5 — Plotting
# ================================================================
px = px_log.cpu().numpy()
pz = pz_log.cpu().numpy()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

for i in range(PLOTS):
    ax[0].plot(px[:, i])
    ax[1].plot(pz[:, i])

ax[0].set_title("Horizontal Position (px) — Deterministic MPC")
ax[0].set_xlabel("Time Step")
ax[0].set_ylabel("px (m)")
ax[0].grid(True)

ax[1].set_title("Vertical Position (pz) — Deterministic MPC")
ax[1].set_xlabel("Time Step")
ax[1].set_ylabel("pz (m)")
ax[1].grid(True)

plt.tight_layout()
plt.savefig("results/mpc_deterministic_trajs.png", dpi=150)
plt.show()

print("Figure saved to results/mpc_deterministic_trajs.png")
