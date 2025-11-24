import os
import casadi as ca
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src import CusadiFunction

# ============================================================
# 1. General settings
# ============================================================
device = "cuda"
dtype = torch.double

NX = 4
NU = 1

HORIZON = 30
NUM_CAND = 128       # Number of candidate sequences
EVAL_BATCH = 4096
STEPS = 500
PLOTS = 10

WIND_STD = 1.5

print("\n=== Batched Deterministic Shooting MPC — CartPole ===")


# ============================================================
# 2. Load CasADi Model + GPU wrapper
# ============================================================
fn_casadi = ca.Function.load("src/casadi_functions/fn_cartpole_sim_step.casadi")

# Batch for deterministic MPC = number of candidate sequences
BATCH_DET = NUM_CAND
fn_gpu_det = CusadiFunction(fn_casadi, BATCH_DET)

# Batch for evaluation rollout
fn_gpu_eval = CusadiFunction(fn_casadi, EVAL_BATCH)


# Physical parameters
m_c  = 1.0
m_p  = 0.3
l    = 0.5
g    = 9.81
dt   = 0.01

def make_params(batch):
    return [
        torch.full((batch,1), m_c, dtype=dtype, device=device),
        torch.full((batch,1), m_p, dtype=dtype, device=device),
        torch.full((batch,1), l, dtype=dtype, device=device),
        torch.full((batch,1), g, dtype=dtype, device=device),
        torch.full((batch,1), dt, dtype=dtype, device=device),
    ]

params_det  = make_params(BATCH_DET)
params_eval = make_params(EVAL_BATCH)


# ============================================================
# 3. Trim point & cost weights
# ============================================================
x_trim = torch.tensor([0., 0., 0., 0.], device=device, dtype=dtype)
u_trim = torch.tensor([0.], device=device, dtype=dtype)

Q = torch.diag(torch.tensor([10., 1., 40., 5.], device=device, dtype=dtype))
R = torch.diag(torch.tensor([0.1], device=device, dtype=dtype))

theta_max = 0.35
track_limit = 1.0


# ============================================================
# 4. GPU step wrapper
# ============================================================
def step_gpu(fn_gpu, x, u, w, params):
    fn_gpu.evaluate([x, u, w, *params])
    return fn_gpu.outputs_sparse[0].clone()


# ============================================================
# 5. Batched deterministic cost evaluation for all candidates
# ============================================================
def evaluate_candidates_det(x0, U_cand):
    """
    x0: (NX,)
    U_cand: (K, N, NU)   candidate sequences
    Returns: cost (K,)
    """
    K, N, nu = U_cand.shape

    # Repeat x0 for all K candidates
    x = x0.view(1, NX).repeat(K, 1)

    cost = torch.zeros(K, device=device, dtype=dtype)

    for t in range(N):
        # Control inputs for all candidates at time t (K, NU)
        u = U_cand[:, t, :]

        # No disturbance (deterministic)
        w = torch.zeros((K, 2), device=device, dtype=dtype)

        # Propagate all K states in parallel
        x = step_gpu(fn_gpu_det, x, u, w, params_det)

        # Compute stage cost (vectorized)
        dx = x - x_trim
        du = u - u_trim

        quad = torch.sum((dx @ Q) * dx, dim=1) + torch.sum((du @ R) * du, dim=1)

        cost += quad

    return cost


# ============================================================
# 6. MPC loop
# ============================================================
def sample_candidates(mean_u, std):
    """
    mean_u: (N, NU)
    returns: (K, N, NU)
    """
    noise = std * torch.randn((NUM_CAND, HORIZON, NU), device=device, dtype=dtype)
    return mean_u.unsqueeze(0) + noise


# Initial evaluation batch
x_eval = torch.zeros((EVAL_BATCH, NX), dtype=dtype, device=device)
x_eval[:,2] = 0.45
x_eval[:,0] = 1.2

theta_violation_curve = []
track_violation_curve = []

px_log = torch.zeros((STEPS, PLOTS), device=device)
theta_log = torch.zeros((STEPS, PLOTS), device=device)

# Warm start
mean_U = u_trim.view(1,NU).repeat(HORIZON,1)
std_U  = 0.5


print("Running batched deterministic MPC...")

for t in range(STEPS):

    # 1. Sample K candidate sequences
    U_cand = sample_candidates(mean_U, std_U)

    # 2. Evaluate all K in parallel
    x0_single = x_eval[0]
    costs = evaluate_candidates_det(x0_single, U_cand)

    # 3. Pick best sequence
    k_best = torch.argmin(costs).item()
    U_best = U_cand[k_best]
    u_t = U_best[0]

    # 4. Warm start next iteration
    mean_U = torch.roll(U_best, shifts=-1, dims=0)
    mean_U[-1] = u_trim

    # 5. Apply to evaluation batch with disturbance
    u_batch = u_t.view(1,NU).repeat(EVAL_BATCH,1)
    w_eval = WIND_STD * torch.randn((EVAL_BATCH,2), dtype=dtype, device=device)

    x_eval = step_gpu(fn_gpu_eval, x_eval, u_batch, w_eval, params_eval)

    # 6. Log constraints
    theta_violation_curve.append((torch.abs(x_eval[:,2]) > theta_max).double().mean().item())
    track_violation_curve.append((torch.abs(x_eval[:,0]) > track_limit).double().mean().item())

    # Logs for plotting
    px_log[t] = x_eval[:PLOTS, 0]
    theta_log[t] = x_eval[:PLOTS, 2]


torch.cuda.synchronize()
print("Batched Deterministic MPC completed.")

# ============================================================
# 5 — Summary
# ============================================================

print("\n================ Chance Constraints Summary ================")
print("Final pole-angle violation prob =", theta_violation_curve[-1])
print("Final track violation prob      =", track_violation_curve[-1])
print("============================================================")


# ============================================================
# 7. Plot results
# ============================================================
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(14,5))
plt.plot(theta_violation_curve, label="theta violation")
plt.plot(track_violation_curve, label="track violation")
plt.axhline(0.05, color="red", linestyle="--")
plt.grid(True)
plt.legend()
plt.title("Deterministic MPC (Batched) — Chance Violations")
plt.savefig("results/cartpole_mpc_det_batched_chance.png")
plt.show()

# Trajectories
px_np = px_log.cpu().numpy()
th_np = theta_log.cpu().numpy()

fig, ax = plt.subplots(1,2,figsize=(13,5))
for i in range(PLOTS):
    ax[0].plot(px_np[:,i])
    ax[1].plot(th_np[:,i])

ax[0].set_title("Cart Position p — Batched Deterministic MPC")
ax[1].set_title("Pole Angle θ — Batched Deterministic MPC")
plt.savefig("results/cartpole_mpc_det_batched_trajs.png")
plt.show()

print("Saved batched deterministic MPC results.")
