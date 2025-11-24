import os
import casadi as ca
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src import CusadiFunction

# ============================================================
# 0. General settings
# ============================================================
device = "cuda"
dtype = torch.double

NX = 6    # [p_x, p_z, theta, v_x, v_z, omega]
NU = 2    # [T1, T2]

EVAL_BATCH = 4096
STEPS      = 500
PLOTS      = 10

# Shared MPC hyperparameters
HORIZON        = 20   # N (used for all methods for fairness)
NUM_CANDIDATES = 128  # K — number of sampled control sequences
NUM_SCENARIOS  = 16   # S — for SAA and CVaR
WIND_STD       = 0.2

# SAA / CVaR penalty + risk weights
LAMBDA_PEN   = 500.0
ALPHA_CVAR   = 0.2     # tail fraction in CVaR
LAMBDA_RISK  = 2.0     # how much we care about CVaR vs mean
ALPHA_CHANCE = 0.05    # reference line for chance plot

# Chance constraint on altitude
z_min = 1.0  # "safe" altitude (same as your 2D SAA/CVaR code)

# ============================================================
# 1. Load CasADi quadrotor model and wrap with CusADi
# ============================================================
CUSADI_FUNCTION_DIR = "src/casadi_functions"
fn_casadi = ca.Function.load(
    os.path.join(CUSADI_FUNCTION_DIR, "fn_quad_sim_step.casadi")
)

# For candidate evaluation (det / saa / cvar), we use smaller batch
BATCH_SAA = NUM_CANDIDATES * NUM_SCENARIOS
fn_gpu_saa  = CusadiFunction(fn_casadi, BATCH_SAA)

# For deterministic single-environment evaluation (det MPC)
fn_gpu_single = CusadiFunction(fn_casadi, 1)

# For external Monte Carlo evaluation: full batch
fn_gpu_eval = CusadiFunction(fn_casadi, EVAL_BATCH)

# ============================================================
# 2. Physical parameters
# ============================================================
m_val  = 1.0
I_val  = 0.05
l_val  = 0.2
g_val  = 9.81
dt_val = 0.01

def make_params(batch, device=device, dtype=dtype):
    m  = torch.full((batch,1), m_val,  dtype=dtype, device=device)
    I  = torch.full((batch,1), I_val,  dtype=dtype, device=device)
    l  = torch.full((batch,1), l_val,  dtype=dtype, device=device)
    g  = torch.full((batch,1), g_val,  dtype=dtype, device=device)
    dt = torch.full((batch,1), dt_val, dtype=dtype, device=device)
    return [m, I, l, g, dt]

params_saa   = make_params(BATCH_SAA)
params_eval  = make_params(EVAL_BATCH)
params_single = make_params(1)

# ============================================================
# 3. Trim point and cost weights
# ============================================================
T_hover   = m_val * g_val
x_trim_np = np.array([0., 1., 0., 0., 0., 0.])
u_trim_np = np.array([T_hover/2, T_hover/2])

x_trim = torch.tensor(x_trim_np, device=device, dtype=dtype)
u_trim = torch.tensor(u_trim_np, device=device, dtype=dtype)

# LQR-like weights (same as your earlier code)
Q = torch.diag(torch.tensor([40., 200., 10., 2., 2., 1.],
                            device=device, dtype=dtype))
R = torch.diag(torch.tensor([0.4, 0.4],
                            device=device, dtype=dtype))

theta_max = 0.35  # tilt bound (only used in penalties for SAA/CVaR)

print("=== 2D Quadrotor MPC Comparison (det / SAA / CVaR) ===")
print("x_trim:", x_trim_np)
print("u_trim:", u_trim_np)

# ============================================================
# 4. GPU step helper
# ============================================================
def step_gpu(fn_gpu, x, u, w, params):
    """
    x: (B,NX)
    u: (B,NU)
    w: (B,2)
    params: [m,I,l,g,dt], each (B,1)
    """
    fn_gpu.evaluate([x, u, w, *params])
    return fn_gpu.outputs_sparse[0].clone()

# ============================================================
# 5. Deterministic MPC (shooting) — helper functions
# ============================================================
def eval_det_candidate(x0, U_seq):
    """
    x0: (NX,) torch (nominal state)
    U_seq: (HORIZON, NU) torch
    returns scalar deterministic cost (no wind)
    """
    x = x0.clone()
    cost = 0.0

    for t in range(HORIZON):
        u = U_seq[t]
        w = torch.zeros((1,2), device=device, dtype=dtype)  # no disturbance

        x = step_gpu(fn_gpu_single,
                     x.view(1, NX),
                     u.view(1, NU),
                     w,
                     params_single)[0]   # (NX,)

        dx = x - x_trim
        du = u - u_trim
        stage_cost = (dx @ Q @ dx) + (du @ R @ du)
        cost += stage_cost

    return cost.item()

def mpc_det_policy(x0, mean_U, std):
    """
    Sample K candidate sequences around mean_U, pick the best
    w.r.t. deterministic cost.
    """
    K = NUM_CANDIDATES
    U_cand = mean_U + std * torch.randn(
        (K, HORIZON, NU), device=device, dtype=dtype
    )

    costs = []
    x0_local = x0.detach()
    for k in range(K):
        c = eval_det_candidate(x0_local, U_cand[k])
        costs.append(c)

    best_idx = int(np.argmin(costs))
    return U_cand[best_idx]

# ============================================================
# 6. SAA and CVaR candidate evaluation
# ============================================================
def evaluate_candidates_saa(x0, U_cand):
    """
    x0: (NX,)
    U_cand: (K, HORIZON, NU)
    returns mean SAA cost (K,)
    """
    K, N, nu = U_cand.shape
    S = NUM_SCENARIOS
    B = K * S

    x = x0.view(1, NX).repeat(B, 1)   # (B,NX)
    cost_env = torch.zeros(B, device=device, dtype=dtype)

    for t in range(N):
        U_t = U_cand[:, t, :]                    # (K,NU)
        U_t = U_t.unsqueeze(1).repeat(1, S, 1)   # (K,S,NU)
        u   = U_t.view(B, nu)                    # (B,NU)

        w = WIND_STD * torch.randn((B, 2), device=device, dtype=dtype)
        x = step_gpu(fn_gpu_saa, x, u, w, params_saa)

        dx = x - x_trim.view(1, NX)
        du = u - u_trim.view(1, NU)
        quad = (dx @ Q * dx).sum(dim=1) + (du @ R * du).sum(dim=1)

        theta = x[:, 2]
        z     = x[:, 1]

        tilt_violation = F.softplus(10.0 * (torch.abs(theta) - theta_max)) / 10.0
        alt_violation  = F.softplus(10.0 * (z_min - z)) / 10.0

        stage_cost = quad + LAMBDA_PEN * (tilt_violation + alt_violation)
        cost_env += stage_cost

    cost_env  = cost_env.view(K, S)
    mean_cost = cost_env.mean(dim=1)  # (K,)
    return mean_cost

def evaluate_candidates_cvar(x0, U_cand):
    """
    x0: (NX,)
    U_cand: (K,N,NU)
    returns total_cost (K,), mean_cost (K,), cvar (K,)
    """
    K, N, nu = U_cand.shape
    S = NUM_SCENARIOS
    B = K * S

    x = x0.view(1, NX).repeat(B, 1)
    traj_cost = torch.zeros(B, device=device, dtype=dtype)

    for t in range(N):
        U_t = U_cand[:, t, :]                    # (K,NU)
        U_t = U_t.unsqueeze(1).repeat(1, S, 1)   # (K,S,NU)
        u   = U_t.view(B, nu)                    # (B,NU)

        w = WIND_STD * torch.randn((B, 2), device=device, dtype=dtype)
        x = step_gpu(fn_gpu_saa, x, u, w, params_saa)

        dx = x - x_trim.view(1, NX)
        du = u - u_trim.view(1, NU)
        quad = (dx @ Q * dx).sum(dim=1) + (du @ R * du).sum(dim=1)

        theta = x[:, 2]
        z     = x[:, 1]
        tilt_violation = F.softplus(10.0 * (torch.abs(theta) - theta_max)) / 10.0
        alt_violation  = F.softplus(10.0 * (z_min - z)) / 10.0

        stage_cost = quad + LAMBDA_PEN * (tilt_violation + alt_violation)
        traj_cost += stage_cost

    traj_cost = traj_cost.view(K, S)
    mean_cost = traj_cost.mean(dim=1)

    # CVaR over scenarios
    sorted_cost, _ = torch.sort(traj_cost, dim=1, descending=True)
    tail_count = max(1, int(np.ceil(ALPHA_CVAR * S)))
    tail_costs = sorted_cost[:, :tail_count]
    cvar = tail_costs.mean(dim=1)

    total_cost = mean_cost + LAMBDA_RISK * (cvar - mean_cost)
    return total_cost, mean_cost, cvar

# ============================================================
# 7. Shared MPC loop runner
# ============================================================
def run_mpc_det():
    print("\n=== Running deterministic MPC (2D) ===")
    x_eval = torch.zeros((EVAL_BATCH, NX), dtype=dtype, device=device)
    x_eval[:,1] = 1.0  # altitude

    # logs
    alt_violation = []
    pz_log = torch.zeros((STEPS, PLOTS), device=device)

    # chance bound
    global z_min
    mean_U = u_trim.view(1, NU).repeat(HORIZON, 1)
    std_U  = 1.0

    for t in range(STEPS):
        best_seq = mpc_det_policy(x_eval[0], mean_U, std_U)
        u_t = best_seq[0]

        mean_U = torch.roll(best_seq, shifts=-1, dims=0)
        mean_U[-1] = u_trim

        u_batch = u_t.view(1,NU).repeat(EVAL_BATCH,1)
        w = WIND_STD * torch.randn((EVAL_BATCH,2), device=device, dtype=dtype)

        x_eval = step_gpu(fn_gpu_eval, x_eval, u_batch, w, params_eval)

        alt_violation.append((x_eval[:,1] < z_min).double().mean().item())
        pz_log[t] = x_eval[:PLOTS,1]

    torch.cuda.synchronize()
    print("Deterministic MPC done.")
    return pz_log.cpu().numpy(), np.array(alt_violation)

def run_mpc_saa():
    print("\n=== Running SAA MPC (2D) ===")
    x_eval = torch.zeros((EVAL_BATCH, NX), dtype=dtype, device=device)
    x_eval[:,1] = 1.0

    alt_violation = []
    pz_log = torch.zeros((STEPS, PLOTS), device=device)

    mean_U = u_trim.view(1, NU).repeat(HORIZON, 1)
    std_U  = 2.0

    for t in range(STEPS):
        # sample candidate sequences
        U_cand = mean_U.view(1,HORIZON,NU) + std_U * torch.randn(
            (NUM_CANDIDATES, HORIZON, NU), device=device, dtype=dtype
        )

        x0_single = x_eval[0].clone()
        mean_cost = evaluate_candidates_saa(x0_single, U_cand)

        k_best = torch.argmin(mean_cost).item()
        U_best = U_cand[k_best]
        u_t    = U_best[0]

        mean_U = torch.roll(U_best, shifts=-1, dims=0)
        mean_U[-1] = u_trim

        u_eval = u_t.view(1,NU).repeat(EVAL_BATCH,1)
        w_eval = WIND_STD * torch.randn((EVAL_BATCH,2), device=device, dtype=dtype)

        x_eval = step_gpu(fn_gpu_eval, x_eval, u_eval, w_eval, params_eval)

        alt_violation.append((x_eval[:,1] < z_min).double().mean().item())
        pz_log[t] = x_eval[:PLOTS,1]

    torch.cuda.synchronize()
    print("SAA MPC done.")
    return pz_log.cpu().numpy(), np.array(alt_violation)

def run_mpc_cvar():
    print("\n=== Running CVaR MPC (2D) ===")
    x_eval = torch.zeros((EVAL_BATCH, NX), dtype=dtype, device=device)
    x_eval[:,1] = 1.0

    alt_violation = []
    pz_log = torch.zeros((STEPS, PLOTS), device=device)

    mean_U = u_trim.view(1, NU).repeat(HORIZON, 1)
    std_U  = 2.0

    for t in range(STEPS):
        U_cand = mean_U.view(1,HORIZON,NU) + std_U * torch.randn(
            (NUM_CANDIDATES, HORIZON, NU), device=device, dtype=dtype
        )

        x0_single = x_eval[0].clone()
        total_cost, mean_cost, cvar_cost = evaluate_candidates_cvar(x0_single, U_cand)

        k_best = torch.argmin(total_cost).item()
        U_best = U_cand[k_best]
        u_t    = U_best[0]

        mean_U = torch.roll(U_best, shifts=-1, dims=0)
        mean_U[-1] = u_trim

        u_eval = u_t.view(1,NU).repeat(EVAL_BATCH,1)
        w_eval = WIND_STD * torch.randn((EVAL_BATCH,2), device=device, dtype=dtype)

        x_eval = step_gpu(fn_gpu_eval, x_eval, u_eval, w_eval, params_eval)

        alt_violation.append((x_eval[:,1] < z_min).double().mean().item())
        pz_log[t] = x_eval[:PLOTS,1]

    torch.cuda.synchronize()
    print("CVaR MPC done.")
    return pz_log.cpu().numpy(), np.array(alt_violation)

# ============================================================
# 8. Run all three controllers
# ============================================================
pz_det, alt_det   = run_mpc_det()
pz_saa, alt_saa   = run_mpc_saa()
pz_cvar, alt_cvar = run_mpc_cvar()

# ============================================================
# 9. Pick "best" trajectory (pz) for each method
#    (best = smallest sum of squared altitude error to z_ref=1.0)
# ============================================================
z_ref = 1.0

def best_traj(pz_log):
    # pz_log: (STEPS, PLOTS)
    err = ((pz_log - z_ref)**2).sum(axis=0)  # (PLOTS,)
    best_idx = np.argmin(err)
    return pz_log[:, best_idx], best_idx

pz_det_best, idx_det   = best_traj(pz_det)
pz_saa_best, idx_saa   = best_traj(pz_saa)
pz_cvar_best, idx_cvar = best_traj(pz_cvar)

print("\nBest trajectory indices among first PLOTS envs:")
print(f"  det  best env index: {idx_det}")
print(f"  SAA  best env index: {idx_saa}")
print(f"  CVaR best env index: {idx_cvar}")

# ============================================================
# 10. Plot best p_z trajectories (same axes, same scale)
# ============================================================
t = np.arange(STEPS)

plt.figure(figsize=(10,5))
plt.plot(t, pz_det_best,  label="Deterministic MPC")
plt.plot(t, pz_saa_best,  label="SAA MPC")
plt.plot(t, pz_cvar_best, label="CVaR MPC")

# unify y-limits
all_pz = np.concatenate([pz_det_best, pz_saa_best, pz_cvar_best])
ymin = all_pz.min() - 0.1
ymax = all_pz.max() + 0.1
plt.ylim([ymin, ymax])

plt.axhline(z_ref, color="k", linestyle="--", linewidth=1, label="z_ref = 1.0")
plt.xlabel("Time step")
plt.ylabel("Altitude p_z (m)")
plt.title("Best altitude trajectory per method (2D quadrotor)")
plt.grid(True)
plt.legend()
os.makedirs("results", exist_ok=True)
plt.savefig("results/compare_best_pz_trajectories.png", dpi=150)
plt.show()

# ============================================================
# 11. Plot altitude violation probability over time (all 3)
# ============================================================
plt.figure(figsize=(10,5))
plt.plot(t, alt_det,   label="Deterministic MPC")
plt.plot(t, alt_saa,   label="SAA MPC")
plt.plot(t, alt_cvar,  label="CVaR MPC")
plt.axhline(ALPHA_CHANCE, color="red", linestyle="--",
            label=f"α (chance) = {ALPHA_CHANCE}")
plt.xlabel("Time step")
plt.ylabel("Altitude violation probability")
plt.title("Altitude chance-constraint violation — all methods")
plt.grid(True)
plt.legend()
plt.savefig("results/compare_alt_violation_curves.png", dpi=150)
plt.show()

print("Saved comparison plots to results/.")
