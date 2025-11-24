import os
import casadi as ca
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src import CusadiFunction

# ==========================
# 1. General settings
# ==========================
device = "cuda"
dtype = torch.double

NX = 12   # [x,y,z,vx,vy,vz,phi,theta,psi,p,q,r]
NU = 4    # [T1,T2,T3,T4]

# MPC + SAA hyperparameters
HORIZON        = 30          # N
NUM_CANDIDATES = 128         # K (control sequences)
NUM_SCENARIOS  = 16          # S (disturbance samples per candidate)
WIND_STD       = 0.2
ALPHA          = 0.05        # for reference in plots

# Outer evaluation batch (for chance curves)
EVAL_BATCH = 4096
STEPS      = 500
PLOTS      = 10

# ==========================
# 2. Load 3D CasADi quadrotor step and wrap with CusADi
# ==========================
CUSADI_FUNCTION_DIR = "src/casadi_functions"
fn_casadi = ca.Function.load(
    os.path.join(CUSADI_FUNCTION_DIR, "fn_quad_sim_step_3d.casadi")
)

# For SAA optimization: batch = NUM_CANDIDATES * NUM_SCENARIOS
BATCH_SAA = NUM_CANDIDATES * NUM_SCENARIOS
fn_gpu_saa  = CusadiFunction(fn_casadi, BATCH_SAA)

# For external evaluation: batch = EVAL_BATCH
fn_gpu_eval = CusadiFunction(fn_casadi, EVAL_BATCH)

# Physical parameters for 3D model
m_val   = 1.0
Jx_val  = 0.02
Jy_val  = 0.02
Jz_val  = 0.04
l_val   = 0.2
k_m_val = 0.01
g_val   = 9.81
dt_val  = 0.01

def make_params(batch, device=device, dtype=dtype):
    m   = torch.full((batch,1), m_val,   dtype=dtype, device=device)
    Jx  = torch.full((batch,1), Jx_val,  dtype=dtype, device=device)
    Jy  = torch.full((batch,1), Jy_val,  dtype=dtype, device=device)
    Jz  = torch.full((batch,1), Jz_val,  dtype=dtype, device=device)
    l   = torch.full((batch,1), l_val,   dtype=dtype, device=device)
    k_m = torch.full((batch,1), k_m_val, dtype=dtype, device=device)
    g   = torch.full((batch,1), g_val,   dtype=dtype, device=device)
    dt  = torch.full((batch,1), dt_val,  dtype=dtype, device=device)
    return [m, Jx, Jy, Jz, l, k_m, g, dt]

params_saa  = make_params(BATCH_SAA)
params_eval = make_params(EVAL_BATCH)

# ==========================
# 3. Trim point and cost weights
# ==========================
T_hover = m_val * g_val
T_each  = T_hover / 4.0

x_trim_np = np.array([
    0., 0., 1.,    # x, y, z
    0., 0., 0.,    # vx, vy, vz
    0., 0., 0.,    # phi, theta, psi
    0., 0., 0.     # p, q, r
], float)

u_trim_np = np.array([T_each, T_each, T_each, T_each], float)

x_trim = torch.tensor(x_trim_np, device=device, dtype=dtype)
u_trim = torch.tensor(u_trim_np, device=device, dtype=dtype)

# Q penalizes [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
Q = torch.diag(torch.tensor(
    [10., 10., 40.,   # x, y, z
     2.,  2.,  5.,    # vx, vy, vz
     20., 20., 5.,    # phi, theta, psi
     1.,  1.,  1.],   # p, q, r
    device=device, dtype=dtype
))

# R penalizes rotor thrusts
R = torch.diag(torch.tensor(
    [0.5, 0.5, 0.5, 0.5],
    device=device, dtype=dtype
))

# 3D chance constraints
tilt_max        = 0.35   # rad, on sqrt(phi^2 + theta^2)
z_min           = 1.0    # minimum safe altitude
lambda_penalty  = 2000.0 # weight on soft penalties

print("=== 3D GPU SAA MPC ===")
print("x_trim:", x_trim_np)
print("u_trim:", u_trim_np)

# ==========================
# 4. Helper: one-step GPU dynamics via CusADi
# ==========================
def step_gpu(fn_gpu, x, u, w, params):
    """
    x: (B, NX), u: (B, NU), w: (B, 3)
    params: list of (B,1) tensors [m,Jx,Jy,Jz,l,k_m,g,dt]
    returns x_next: (B, NX)
    """
    fn_gpu.evaluate([x, u, w, *params])
    return fn_gpu.outputs_sparse[0].clone()

# ==========================
# 5. SAA evaluation of candidate control sequences
# ==========================
def evaluate_candidates_saa(x0, U_cand):
    """
    x0: (NX,) torch tensor
    U_cand: (K, N, NU) candidates, on GPU
    returns: mean_cost (K,) for each candidate
    """
    K, N, nu = U_cand.shape
    S = NUM_SCENARIOS
    B = K * S

    # Repeat initial state for all candidate-scenario pairs
    x = x0.view(1, NX).repeat(B, 1)   # (B, NX)

    # Accumulate per-env costs
    cost_env = torch.zeros(B, device=device, dtype=dtype)

    for t in range(N):
        # Controls at time t for all (k,s)
        U_t = U_cand[:, t, :]           # (K, NU)
        U_t = U_t.unsqueeze(1).repeat(1, S, 1)  # (K, S, NU)
        u   = U_t.view(B, nu)           # (B, NU)

        # Disturbances at time t
        w = WIND_STD * torch.randn((B, 3), device=device, dtype=dtype)

        # Propagate dynamics
        x = step_gpu(fn_gpu_saa, x, u, w, params_saa)

        # State and control deviations
        dx = x - x_trim.view(1, NX)
        du = u - u_trim.view(1, NU)

        # Quadratic cost
        quad = (dx @ Q * dx).sum(dim=1) + (du @ R * du).sum(dim=1)

        # Soft chance penalties in 3D
        phi   = x[:, 6]
        theta = x[:, 7]
        z     = x[:, 2]

        tilt_angle = torch.sqrt(phi**2 + theta**2)
        tilt_violation = F.softplus(10.0 * (tilt_angle - tilt_max)) / 10.0
        alt_violation  = F.softplus(10.0 * (z_min - z)) / 10.0

        stage_cost = quad + lambda_penalty * (tilt_violation + alt_violation)
        cost_env  += stage_cost

    # Reshape and average over scenarios
    cost_env  = cost_env.view(K, S)
    mean_cost = cost_env.mean(dim=1)  # (K,)
    return mean_cost

# ==========================
# 6. MPC loop with SAA shooting
# ==========================
def sample_candidates(mean_u, std_u):
    """
    mean_u: (N, NU)
    returns U_cand: (K, N, NU)
    """
    K = NUM_CANDIDATES
    noise = std_u * torch.randn((K, HORIZON, NU), device=device, dtype=dtype)
    return mean_u.view(1, HORIZON, NU) + noise

# Initial state for evaluation batch
x_eval = torch.zeros((EVAL_BATCH, NX), dtype=dtype, device=device)
x_eval[:, 2] = 1.0  # initial altitude z = 1

# small horizontal perturbations
x_eval[:, 0] = 0.1 * torch.randn(EVAL_BATCH, device=device, dtype=dtype)
x_eval[:, 1] = 0.1 * torch.randn(EVAL_BATCH, device=device, dtype=dtype)

tilt_violation_curve = []
alt_violation_curve  = []

px_log  = torch.zeros((STEPS, PLOTS), device=device)
py_log  = torch.zeros((STEPS, PLOTS), device=device)
pz_log  = torch.zeros((STEPS, PLOTS), device=device)

# Initial mean control sequence: hover thrust
mean_U = u_trim.view(1, NU).repeat(HORIZON, 1)  # (N, NU)
std_U  = 2.0  # sampling std

for t in range(STEPS):
    # 1) Build candidate sequences around mean_U
    U_cand = sample_candidates(mean_U, std_U)   # (K, N, NU)

    # 2) Evaluate SAA costs on GPU
    x0_single = x_eval[0].clone()              # nominal env
    mean_cost = evaluate_candidates_saa(x0_single, U_cand)

    # 3) Pick best candidate
    k_best = torch.argmin(mean_cost).item()
    U_best = U_cand[k_best]                    # (N, NU)
    u_t    = U_best[0]                         # first control

    # 4) Receding-horizon warm start
    mean_U = torch.roll(U_best, shifts=-1, dims=0)
    mean_U[-1] = u_trim

    # 5) Apply chosen u_t to evaluation batch under fresh noise
    u_eval = u_t.view(1, NU).repeat(EVAL_BATCH, 1)
    w_eval = WIND_STD * torch.randn((EVAL_BATCH, 3),
                                    dtype=dtype, device=device)

    x_eval = step_gpu(fn_gpu_eval, x_eval, u_eval, w_eval, params_eval)

    # 6) Log chance violations
    phi   = x_eval[:, 6]
    theta = x_eval[:, 7]
    z     = x_eval[:, 2]

    tilt_angle = torch.sqrt(phi**2 + theta**2)
    tilt_violation_curve.append((tilt_angle > tilt_max).double().mean().item())
    alt_violation_curve.append((z < z_min).double().mean().item())

    # Log trajectories
    px_log[t] = x_eval[:PLOTS, 0]
    py_log[t] = x_eval[:PLOTS, 1]
    pz_log[t] = x_eval[:PLOTS, 2]

torch.cuda.synchronize()
print("3D SAA MPC simulation complete!")

print("\n================ 3D GPU SAA MPC Summary ================")
print(f"Final tilt violation prob = {tilt_violation_curve[-1]:.4f}")
print(f"Final alt violation prob  = {alt_violation_curve[-1]:.4f}")
print("========================================================")

# ==========================
# 7. Plot chance violations
# ==========================
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(14,5))
plt.plot(tilt_violation_curve, label="Tilt violation probability")
plt.plot(alt_violation_curve, label="Altitude violation probability")
plt.axhline(ALPHA, color="red", linestyle="--", label=f"α = {ALPHA}")
plt.xlabel("Time Step")
plt.ylabel("Probability")
plt.title(
    f"3D Chance Constraint Violations Over Time — GPU SAA MPC "
    f"(K={NUM_CANDIDATES}, S={NUM_SCENARIOS})"
)
plt.grid(True)
plt.legend()
plt.savefig("results/mpc_3d_chance_saa_gpu.png", dpi=150)
plt.show()

# ==========================
# 8. Plot trajectories
# ==========================
px = px_log.cpu().numpy()
py = py_log.cpu().numpy()
pz = pz_log.cpu().numpy()

fig, ax = plt.subplots(1,3, figsize=(15,5))
for i in range(PLOTS):
    ax[0].plot(px[:, i])
    ax[1].plot(py[:, i])
    ax[2].plot(pz[:, i])

ax[0].set_title("x trajectories — GPU SAA MPC")
ax[0].set_xlabel("Time Step")
ax[0].set_ylabel("x (m)")
ax[0].grid(True)

ax[1].set_title("y trajectories — GPU SAA MPC")
ax[1].set_xlabel("Time Step")
ax[1].set_ylabel("y (m)")
ax[1].grid(True)

ax[2].set_title("z trajectories — GPU SAA MPC")
ax[2].set_xlabel("Time Step")
ax[2].set_ylabel("z (m)")
ax[2].grid(True)

plt.tight_layout()
plt.savefig("results/mpc_3d_chance_saa_gpu_trajs.png", dpi=150)
plt.show()
print("Saved 3D SAA MPC plots to results/ directory.")
