import os
import casadi as ca
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src import CusadiFunction

# ================================================================
# 1. General settings
# ================================================================
device = "cuda"
dtype = torch.double

NX = 6
NU = 2

# MPC + SAA hyperparameters
HORIZON        = 20     # N
NUM_CANDIDATES = 128    # K: number of control sequences
NUM_SCENARIOS  = 16     # S: disturbance samples per candidate
WIND_STD       = 0.2

# CVaR hyperparameters
ALPHA_CVAR   = 0.2      # CVaR tail level (worst 20% of scenario costs)
LAMBDA_RISK  = 2.0      # how much we care about CVaR vs mean
LAMBDA_PEN   = 2000.0   # weight on soft constraint penalties inside stage cost

# Chance-constraint reference line (for plotting only)
ALPHA_CHANCE = 0.05

# Monte Carlo evaluation batch
EVAL_BATCH = 4096
STEPS      = 500
PLOTS      = 10

# ================================================================
# 2. Load CasADi quadrotor step, wrap with CusADi
# ================================================================
CUSADI_FUNCTION_DIR = "src/casadi_functions"
fn_casadi = ca.Function.load(
    os.path.join(CUSADI_FUNCTION_DIR, "fn_quad_sim_step.casadi")
)

# For SAA optimization: batch = K * S
BATCH_SAA = NUM_CANDIDATES * NUM_SCENARIOS
fn_gpu_saa = CusadiFunction(fn_casadi, BATCH_SAA)

# For external evaluation: batch = EVAL_BATCH
fn_gpu_eval = CusadiFunction(fn_casadi, EVAL_BATCH)

# Physical parameters
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

params_saa  = make_params(BATCH_SAA)
params_eval = make_params(EVAL_BATCH)

# ================================================================
# 3. Trim point and cost weights
# ================================================================
T_hover   = m_val * g_val
x_trim_np = np.array([0., 1., 0., 0., 0., 0.])
u_trim_np = np.array([T_hover/2, T_hover/2])

x_trim = torch.tensor(x_trim_np, device=device, dtype=dtype)
u_trim = torch.tensor(u_trim_np, device=device, dtype=dtype)

Q = torch.diag(torch.tensor([40., 40., 10., 2., 2., 1.],
                            device=device, dtype=dtype))
R = torch.diag(torch.tensor([0.4, 0.4],
                            device=device, dtype=dtype))

theta_max = 0.35
z_min     = 1.0

print("=== CVaR MPC (GPU SAA) ===")
print("x_trim:", x_trim_np)
print("u_trim:", u_trim_np)

# ================================================================
# 4. One-step GPU dynamics helper
# ================================================================
def step_gpu(fn_gpu, x, u, w, params):
    """
    x: (B, NX)
    u: (B, NU)
    w: (B, 2)
    params: [m, I, l, g, dt] each (B,1)
    returns x_next: (B, NX)
    """
    fn_gpu.evaluate([x, u, w, *params])
    return fn_gpu.outputs_sparse[0].clone()

# ================================================================
# 5. CVaR-based evaluation of candidate control sequences
# ================================================================
def evaluate_candidates_cvar(x0, U_cand):
    """
    x0:      (NX,) torch tensor — current nominal state
    U_cand:  (K, N, NU) — candidate control sequences (on GPU)
    returns: total_cost (K,) — CVaR-based cost for each candidate
    """
    K, N, nu = U_cand.shape
    S        = NUM_SCENARIOS
    B        = K * S

    # Repeat initial state across all candidate-scenario pairs
    x = x0.view(1, NX).repeat(B, 1)   # (B, NX)

    # Accumulate per-env trajectory cost
    traj_cost = torch.zeros(B, device=device, dtype=dtype)

    for t in range(N):
        # Controls at time t for all (k, s)
        U_t = U_cand[:, t, :]               # (K, NU)
        U_t = U_t.unsqueeze(1).repeat(1, S, 1)   # (K, S, NU)
        u   = U_t.view(B, nu)               # (B, NU)

        # Disturbances at time t for all envs
        w = WIND_STD * torch.randn((B, 2), device=device, dtype=dtype)

        # Propagate dynamics
        x = step_gpu(fn_gpu_saa, x, u, w, params_saa)

        # Quadratic tracking cost
        dx = x - x_trim.view(1, NX)
        du = u - u_trim.view(1, NU)

        quad = (dx @ Q * dx).sum(dim=1) + (du @ R * du).sum(dim=1)

        # Soft chance penalties
        theta = x[:, 2]
        z     = x[:, 1]

        tilt_violation = F.softplus(10.0 * (torch.abs(theta) - theta_max)) / 10.0
        alt_violation  = F.softplus(10.0 * (z_min - z)) / 10.0

        stage_cost = quad + LAMBDA_PEN * (tilt_violation + alt_violation)

        traj_cost += stage_cost

    # Reshape to (K, S)
    traj_cost = traj_cost.view(K, S)

    # Mean cost across scenarios (risk-neutral part)
    mean_cost = traj_cost.mean(dim=1)   # (K,)

    # CVaR_α across scenarios (tail risk)
    # Sort each row descending and average worst α fraction
    sorted_cost, _ = torch.sort(traj_cost, dim=1, descending=True)
    tail_count = max(1, int(np.ceil(ALPHA_CVAR * S)))
    tail_costs = sorted_cost[:, :tail_count]
    cvar = tail_costs.mean(dim=1)       # (K,)

    # Total CVaR-MPC objective for each candidate
    total_cost = mean_cost + LAMBDA_RISK * (cvar - mean_cost)
    return total_cost, mean_cost, cvar

# ================================================================
# 6. MPC loop (random shooting with CVaR objective)
# ================================================================
def sample_candidates(mean_u, std_u):
    """
    mean_u: (N, NU) — base mean control sequence
    returns: (K, N, NU) candidate sequences
    """
    K = NUM_CANDIDATES
    noise = std_u * torch.randn((K, HORIZON, NU), device=device, dtype=dtype)
    return mean_u.view(1, HORIZON, NU) + noise

# Initial state for evaluation batch
x_eval = torch.zeros((EVAL_BATCH, NX), dtype=dtype, device=device)
x_eval[:, 1] = 1.0  # initial altitude

tilt_violation_curve = []
alt_violation_curve  = []

px_log = torch.zeros((STEPS, PLOTS), device=device)
pz_log = torch.zeros((STEPS, PLOTS), device=device)

# Initial mean control sequence around hover
mean_U = u_trim.view(1, NU).repeat(HORIZON, 1)   # (N, NU)
std_U  = 2.0                                     # sampling std

for t in range(STEPS):

    # 1) Sample candidate sequences around current mean_U
    U_cand = sample_candidates(mean_U, std_U)    # (K, N, NU)

    # 2) Evaluate CVaR objective for each candidate on GPU
    x0_single = x_eval[0].clone()
    total_cost, mean_cost, cvar_cost = evaluate_candidates_cvar(x0_single, U_cand)

    # 3) Pick the best candidate (lowest CVaR-based cost)
    k_best = torch.argmin(total_cost).item()
    U_best = U_cand[k_best]                      # (N, NU)
    u_t    = U_best[0]                           # first action

    # Optional: receding-horizon warm start → shift U_best
    mean_U = torch.roll(U_best, shifts=-1, dims=0)
    mean_U[-1] = u_trim   # last step guess back to hover

    # 4) Apply u_t to evaluation batch with fresh disturbances
    u_eval = u_t.view(1, NU).repeat(EVAL_BATCH, 1)
    w_eval = WIND_STD * torch.randn((EVAL_BATCH, 2),
                                    dtype=dtype, device=device)

    x_eval = step_gpu(fn_gpu_eval, x_eval, u_eval, w_eval, params_eval)

    # 5) Log chance violations (over eval batch)
    tilt_violation_curve.append(
        (torch.abs(x_eval[:,2]) > theta_max).double().mean().item()
    )
    alt_violation_curve.append(
        (x_eval[:,1] < z_min).double().mean().item()
    )

    # 6) Log sample trajectories
    px_log[t] = x_eval[:PLOTS, 0]
    pz_log[t] = x_eval[:PLOTS, 1]

torch.cuda.synchronize()
print("CVaR MPC simulation complete!")

print("\n================ CVaR MPC Summary ================")
print(f"Final tilt violation prob = {tilt_violation_curve[-1]:.4f}")
print(f"Final alt  violation prob = {alt_violation_curve[-1]:.4f}")
print("==================================================")

# ================================================================
# 7. Plot chance violations
# ================================================================
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(14,5))
plt.plot(tilt_violation_curve, label="Tilt violation probability")
plt.plot(alt_violation_curve, label="Altitude violation probability")
plt.axhline(ALPHA_CHANCE, color="red", linestyle="--",
            label=f"α (chance) = {ALPHA_CHANCE}")
plt.xlabel("Time Step")
plt.ylabel("Probability")
plt.title(f"Chance Constraint Violations Over Time — CVaR MPC "
          f"(K={NUM_CANDIDATES}, S={NUM_SCENARIOS}, α_CVaR={ALPHA_CVAR})")
plt.grid(True)
plt.legend()
plt.savefig("results/mpc_cvar_chance.png", dpi=150)
plt.show()

# ================================================================
# 8. Plot sample trajectories
# ================================================================
px = px_log.cpu().numpy()
pz = pz_log.cpu().numpy()

fig, ax = plt.subplots(1,2, figsize=(13,5))
for i in range(PLOTS):
    ax[0].plot(px[:, i])
    ax[1].plot(pz[:, i])

ax[0].set_title("Horizontal Position — CVaR MPC")
ax[0].set_xlabel("Time Step")
ax[0].set_ylabel("px (m)")
ax[0].grid(True)

ax[1].set_title("Vertical Position — CVaR MPC")
ax[1].set_xlabel("Time Step")
ax[1].set_ylabel("pz (m)")
ax[1].grid(True)

plt.tight_layout()
plt.savefig("results/mpc_cvar_trajs.png", dpi=150)
plt.show()

print("Saved CVaR MPC plots to results/.")
