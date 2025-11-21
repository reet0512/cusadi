import casadi as ca
import numpy as np
import torch
import matplotlib.pyplot as plt
from src import CusadiFunction

# ================================================================
# SECTION 1 — Load Discrete Quadrotor Model
# ================================================================
fn = ca.Function.load("src/casadi_functions/fn_quad_sim_step.casadi")

nx = 6
nu = 2

# physical params
m  = 1.0
I  = 0.05
l  = 0.2
g  = 9.81
dt = 0.01

params_sym = [ca.MX(m), ca.MX(I), ca.MX(l), ca.MX(g), ca.MX(dt)]

# ================================================================
# SECTION 2 — Trim
# ================================================================
T_hover = m * g
u_trim = np.array([T_hover/2, T_hover/2])
x_trim = np.array([0., 1., 0., 0., 0., 0.])

print("=== SAA MPC Trim ===")
print("x_trim:", x_trim)
print("u_trim:", u_trim)

# ================================================================
# SECTION 3 — Full SAA MPC Inside CasADi
# ================================================================
N = 30      # horizon
M = 32      # SAA samples
wind_std = 0.2
alpha = 0.05

np.random.seed(123)
w_saa = wind_std * np.random.randn(M, N, 2)

# decision variables
U = ca.MX.sym("U", N, nu)
x0_param = ca.MX.sym("x0", nx)

# cost matrices
Q = np.diag([40, 40, 10, 2, 2, 1])
R = np.diag([0.4, 0.4])

Q_c = ca.DM(Q)
R_c = ca.DM(R)

def softplus(z):
    return ca.log(1 + ca.exp(z))

# cost
cost = 0
g_con = []

# -----------------------------
# Nominal trajectory cost only
# -----------------------------
x_nom = x0_param
for k in range(N):
    uk = U[k, :].reshape((nu,1))
    x_nom = fn(x_nom, uk, ca.DM.zeros(2), *params_sym)

    dx = x_nom - x_trim
    du = uk - u_trim

    cost += ca.mtimes([dx.T, Q_c, dx]) + ca.mtimes([du.T, R_c, du])

# ------------------------------------------
# SAA penalty terms
# ------------------------------------------
for j in range(M):
    xj = x0_param
    for k in range(N):
        uk = U[k,:].reshape((nu,1))
        wj = ca.DM(w_saa[j,k])

        xj = fn(xj, uk, wj, *params_sym)

        theta = xj[2]
        z     = xj[1]

        tilt_violation = softplus(10*(ca.fabs(theta) - 0.35)) / 10
        alt_violation  = softplus(10*(1.0 - z)) / 10

        cost += (4000.0 / M) * (tilt_violation + alt_violation)

# ------------------------------------------
# Build NLP
# ------------------------------------------
nlp = {
    "x": ca.vec(U),
    "f": cost,
    "g": ca.vertcat(*g_con),
    "p": x0_param
}

solver = ca.nlpsol("mpc", "ipopt", nlp,
    {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 700})

print("\n=== SAA MPC Solver Built ===")

# ================================================================
# SECTION 4 — External Monte Carlo Evaluation (Batch=4096)
# ================================================================
device = "cuda"
dtype = torch.double

BATCH = 4096
STEPS = 500
PLOTS = 10

fn_gpu = CusadiFunction(fn, BATCH)

x_eval = torch.zeros((BATCH, nx), dtype=dtype, device=device)
x_eval[:,1] = 1.0

tilt_violation = []
alt_violation = []

px_log = torch.zeros((STEPS, PLOTS), device=device)
pz_log = torch.zeros((STEPS, PLOTS), device=device)

for t in range(STEPS):

    U_guess = np.tile(u_trim, (N,1))

    sol = solver(
        x0 = U_guess.reshape(-1,1),
        p  = x_eval[0].cpu().numpy().reshape(-1,1)
    )

    U_opt = sol["x"].full().reshape(N,nu)
    u_act = torch.tensor(U_opt[0], dtype=dtype, device=device).repeat(BATCH,1)

    # rollout
    w = wind_std * torch.randn((BATCH,2), dtype=dtype, device=device)

    params = [
        torch.full((BATCH,1), m, dtype=dtype, device=device),
        torch.full((BATCH,1), I, dtype=dtype, device=device),
        torch.full((BATCH,1), l, dtype=dtype, device=device),
        torch.full((BATCH,1), g, dtype=dtype, device=device),
        torch.full((BATCH,1), dt, dtype=dtype, device=device),
    ]

    fn_gpu.evaluate([x_eval, u_act, w, *params])
    x_eval = fn_gpu.outputs_sparse[0].clone()

    px_log[t] = x_eval[:PLOTS,0]
    pz_log[t] = x_eval[:PLOTS,1]

    tilt_violation.append((torch.abs(x_eval[:,2]) > 0.35).float().mean().item())
    alt_violation.append((x_eval[:,1] < 1.0).float().mean().item())

torch.cuda.synchronize()
print("Simulation complete!")

print("\n================ SAA MPC Summary ================")
print(f"Final tilt violation prob = {tilt_violation[-1]:.4f}")
print(f"Final alt violation prob  = {alt_violation[-1]:.4f}")
print("=================================================")

# ================================================================
# PLOTS (same as LQR baseline)
# ================================================================
plt.figure(figsize=(14,5))
plt.plot(tilt_violation, label="Tilt violation probability")
plt.plot(alt_violation, label="Altitude violation probability")
plt.axhline(alpha, color="red", linestyle="--", label=f"α = {alpha}")
plt.xlabel("Time Step")
plt.ylabel("Probability")
plt.title("Chance-Constrained SAA MPC (32 Scenarios)")
plt.grid(True)
plt.legend()
plt.savefig("results/mpc_chance_saa.png", dpi=150)
plt.show()

# ================================================================
# Trajectory plots
# ================================================================
px = px_log.cpu().numpy()
pz = pz_log.cpu().numpy()

fig, ax = plt.subplots(1,2, figsize=(13,5))
for i in range(PLOTS):
    ax[0].plot(px[:,i])
    ax[1].plot(pz[:,i])

ax[0].set_title("Horizontal Position — SAA MPC")
ax[0].set_xlabel("Time Step")
ax[0].set_ylabel("px (m)")
ax[0].grid(True)

ax[1].set_title("Vertical Position — SAA MPC")
ax[1].set_xlabel("Time Step")
ax[1].set_ylabel("pz (m)")
ax[1].grid(True)

plt.tight_layout()
plt.savefig("results/mpc_chance_saa_trajs.png", dpi=150)
plt.show()

print("Saved plots to results/ directory.")
