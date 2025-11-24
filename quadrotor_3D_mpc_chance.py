import os
import casadi as ca
import numpy as np
import torch
import matplotlib.pyplot as plt
from src import CusadiFunction

# ============================================================
# 1. General settings
# ============================================================
device = "cuda"
dtype = torch.double

NX = 12   # [x,y,z,vx,vy,vz,phi,theta,psi,p,q,r]
NU = 4    # [T1,T2,T3,T4]

HORIZON    = 30          # planning horizon
WIND_STD   = 0.2
EVAL_BATCH = 4096
STEPS      = 500
PLOTS      = 10

# ============================================================
# 2. Load 3D model
# ============================================================
CUSADI_FUNCTION_DIR = "src/casadi_functions"
fn_casadi = ca.Function.load(
    os.path.join(CUSADI_FUNCTION_DIR, "fn_quad_sim_step_3d.casadi")
)

# One CusADi function for evaluation batch
fn_gpu_eval = CusadiFunction(fn_casadi, EVAL_BATCH)
# Another for single-environment candidate evaluation
fn_gpu_single = CusadiFunction(fn_casadi, 1)

# physical params
m_val   = 1.0
Jx_val  = 0.02
Jy_val  = 0.02
Jz_val  = 0.04
l_val   = 0.2
k_m_val = 0.01
g_val   = 9.81
dt_val  = 0.01

def make_params(batch):
    return [
        torch.full((batch,1), m_val,   dtype=dtype, device=device),
        torch.full((batch,1), Jx_val,  dtype=dtype, device=device),
        torch.full((batch,1), Jy_val,  dtype=dtype, device=device),
        torch.full((batch,1), Jz_val,  dtype=dtype, device=device),
        torch.full((batch,1), l_val,   dtype=dtype, device=device),
        torch.full((batch,1), k_m_val, dtype=dtype, device=device),
        torch.full((batch,1), g_val,   dtype=dtype, device=device),
        torch.full((batch,1), dt_val,  dtype=dtype, device=device),
    ]

params_eval   = make_params(EVAL_BATCH)
params_single = make_params(1)

# ============================================================
# 3. Trim + cost weights
# ============================================================
T_hover = m_val * g_val
T_each  = T_hover / 4.0

x_trim_np = np.array([
    0., 0., 1.,   # x, y, z
    0., 0., 0.,   # vx, vy, vz
    0., 0., 0.,   # phi, theta, psi
    0., 0., 0.    # p, q, r
], float)

u_trim_np = np.array([T_each, T_each, T_each, T_each], float)

x_trim = torch.tensor(x_trim_np, device=device, dtype=dtype)
u_trim = torch.tensor(u_trim_np, device=device, dtype=dtype)

# Q penalizes [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
Q = torch.diag(torch.tensor(
    [10., 10., 40.,   # position
     2.,  2.,  5.,    # velocity
     20., 20., 5.,    # roll, pitch, yaw
     1.,  1.,  1.],   # p, q, r
    device=device, dtype=dtype
))

# R penalizes rotor thrusts
R = torch.diag(torch.tensor(
    [0.5, 0.5, 0.5, 0.5],
    device=device, dtype=dtype
))

print("=== 3D Deterministic Shooting MPC (Chance Evaluation) ===")
print("Trim state:", x_trim_np)
print("Trim input:", u_trim_np)

# ============================================================
# 4. GPU step wrapper
# ============================================================
def step_gpu(fn_gpu, x, u, w, params):
    """
    x: (B,NX)
    u: (B,NU)
    w: (B,3)
    params: list of tensors [m,Jx,Jy,Jz,l,k_m,g,dt] each (B,1)
    """
    fn_gpu.evaluate([x, u, w, *params])
    return fn_gpu.outputs_sparse[0].clone()

# ============================================================
# 5. Evaluate a SINGLE candidate sequence (deterministic)
# ============================================================
def eval_det_candidate(x0, U_seq):
    """
    x0: (NX,) torch
    U_seq: (HORIZON, NU) torch
    returns scalar deterministic cost (no wind)
    """
    x = x0.clone()
    cost = 0.0

    for t in range(HORIZON):
        u = U_seq[t]  # (NU,)

        # no disturbance during candidate evaluation
        w = torch.zeros((1,3), device=device, dtype=dtype)

        x = step_gpu(
            fn_gpu_single,
            x.view(1, NX),
            u.view(1, NU),
            w,
            params_single
        )[0]  # get back (NX,)

        dx = x - x_trim
        du = u - u_trim

        stage_cost = (dx @ Q @ dx) + (du @ R @ du)
        cost += stage_cost

    return cost.item()

# ============================================================
# 6. Deterministic MPC Policy (shooting)
# ============================================================
def mpc_det_policy(x0, mean_U, std):
    """
    One-step deterministic shooting MPC:
      - Sample K candidate sequences around mean_U
      - Evaluate each with eval_det_candidate (no wind)
      - Return the best sequence U* of shape (HORIZON, NU)
    """
    K = 128  # number of candidate control sequences

    # U_cand: (K, HORIZON, NU)
    U_cand = mean_U + std * torch.randn(
        (K, HORIZON, NU), device=device, dtype=dtype
    )

    costs = []
    x0_local = x0.detach()  # avoid weird grads

    for k in range(K):
        c = eval_det_candidate(x0_local, U_cand[k])
        costs.append(c)

    best_idx = int(np.argmin(costs))
    return U_cand[best_idx]

# ============================================================
# 7. Evaluate full MPC performance over 4096 environments
# ============================================================
x_eval = torch.zeros((EVAL_BATCH, NX), dtype=dtype, device=device)
x_eval[:,2] = 1.0  # initial altitude z ~ 1.0

# small initial horizontal perturbation
x_eval[:,0] = 0.1 * torch.randn(EVAL_BATCH, dtype=dtype, device=device)
x_eval[:,1] = 0.1 * torch.randn(EVAL_BATCH, dtype=dtype, device=device)

tilt_violation_curve = []
alt_violation_curve  = []

px_log    = torch.zeros((STEPS, PLOTS), device=device)
py_log    = torch.zeros((STEPS, PLOTS), device=device)
pz_log    = torch.zeros((STEPS, PLOTS), device=device)
phi_log   = torch.zeros((STEPS, PLOTS), device=device)
theta_log = torch.zeros((STEPS, PLOTS), device=device)

# chance constraint thresholds
tilt_max = 0.35  # rad
z_min    = 1.0   # m

# initial warm-start control sequence
mean_U = u_trim.view(1, NU).repeat(HORIZON, 1)
std_U  = 1.0

print("Running 3D deterministic shooting MPC with chance evaluation...")

for t in range(STEPS):
    # 1) Choose deterministic best sequence for the first environment
    best_seq = mpc_det_policy(x_eval[0], mean_U, std_U)
    u_t = best_seq[0]  # first control in the sequence: (NU,)

    # 2) Warm start next iteration by shifting the best sequence
    mean_U = torch.roll(best_seq, shifts=-1, dims=0)
    mean_U[-1] = u_trim  # last step goes back toward trim

    # 3) Apply control to evaluation batch (with wind)
    u_batch = u_t.view(1,NU).repeat(EVAL_BATCH, 1)
    w = WIND_STD * torch.randn((EVAL_BATCH, 3), device=device, dtype=dtype)

    x_eval = step_gpu(fn_gpu_eval, x_eval, u_batch, w, params_eval)

    # 4) Log chance constraint violations
    # tilt = sqrt(phi^2 + theta^2)
    tilt_angle = torch.sqrt(x_eval[:,6]**2 + x_eval[:,7]**2)
    tilt_violation_curve.append(
        (tilt_angle > tilt_max).double().mean().item()
    )
    alt_violation_curve.append(
        (x_eval[:,2] < z_min).double().mean().item()
    )

    # 5) Log sample trajectories
    px_log[t]    = x_eval[:PLOTS, 0]
    py_log[t]    = x_eval[:PLOTS, 1]
    pz_log[t]    = x_eval[:PLOTS, 2]
    phi_log[t]   = x_eval[:PLOTS, 6]
    theta_log[t] = x_eval[:PLOTS, 7]

torch.cuda.synchronize()
print("Deterministic 3D shooting MPC done.")

print("\n=== Summary (3D MPC Chance) ===")
print("Final tilt prob:", tilt_violation_curve[-1])
print("Final alt  prob:", alt_violation_curve[-1])

# ============================================================
# 8. Chance violation plots
# ============================================================
plt.figure(figsize=(14,5))
plt.plot(tilt_violation_curve, label="Tilt violation prob")
plt.plot(alt_violation_curve,  label="Alt violation prob")
plt.axhline(0.05, color="red", linestyle="--", label="α = 0.05")
plt.grid(True)
plt.legend()
plt.title("3D Deterministic Shooting MPC — Chance Evaluation")
os.makedirs("results", exist_ok=True)
plt.savefig("results/mpc_3d_det_shooting_chance.png", dpi=150)
plt.show()

# ============================================================
# 9. Trajectory plots (x,y,z)
# ============================================================
px = px_log.cpu().numpy()
py = py_log.cpu().numpy()
pz = pz_log.cpu().numpy()

fig,ax = plt.subplots(1,3,figsize=(15,4))
for i in range(PLOTS):
    ax[0].plot(px[:,i])
    ax[1].plot(py[:,i])
    ax[2].plot(pz[:,i])

ax[0].set_title("x trajectories")
ax[0].set_xlabel("Time step")
ax[0].set_ylabel("x (m)")

ax[1].set_title("y trajectories")
ax[1].set_xlabel("Time step")
ax[1].set_ylabel("y (m)")

ax[2].set_title("z trajectories")
ax[2].set_xlabel("Time step")
ax[2].set_ylabel("z (m)")

plt.tight_layout()
plt.savefig("results/mpc_3d_det_shooting_trajs.png", dpi=150)
plt.show()

print("Saved deterministic 3D shooting MPC results.")
