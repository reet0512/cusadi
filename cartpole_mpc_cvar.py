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

NX = 4
NU = 1

# MPC + SAA parameters
HORIZON        = 20
NUM_CANDIDATES = 128
NUM_SCENARIOS  = 16
WIND_STD       = 1.5

# CVaR parameters
ALPHA_CVAR  = 0.1     # tail fraction (worst 10%)
LAMBDA_RISK = 0.5     # weight on CVaR vs mean
LAMBDA_PEN  = 2000.0  # soft-penalty weight for violations

# Eval
EVAL_BATCH = 4096
STEPS = 500
PLOTS = 10

print("\n=== CVaR MPC — CartPole ===")

# ================================================================
# 2. Load CasADi sim + GPU wrappers
# ================================================================
fn_casadi = ca.Function.load("src/casadi_functions/fn_cartpole_sim_step.casadi")

BATCH_SAA = NUM_CANDIDATES * NUM_SCENARIOS
fn_gpu_saa  = CusadiFunction(fn_casadi, BATCH_SAA)
fn_gpu_eval = CusadiFunction(fn_casadi, EVAL_BATCH)

# physical params
m_c  = 1.0
m_p  = 0.3
l    = 0.5
g    = 9.81
dt   = 0.01

def make_params(batch):
    return [
        torch.full((batch,1), m_c, dtype=dtype, device=device),
        torch.full((batch,1), m_p, dtype=dtype, device=device),
        torch.full((batch,1), l,   dtype=dtype, device=device),
        torch.full((batch,1), g,   dtype=dtype, device=device),
        torch.full((batch,1), dt,  dtype=dtype, device=device),
    ]

params_saa  = make_params(BATCH_SAA)
params_eval = make_params(EVAL_BATCH)

# ================================================================
# 3. Trim state + cost weights + constraints
# ================================================================
x_trim = torch.tensor([0.,0.,0.,0.], device=device, dtype=dtype)
u_trim = torch.tensor([0.], device=device, dtype=dtype)

Q = torch.diag(torch.tensor([10., 1., 40., 5.], device=device, dtype=dtype))
R = torch.diag(torch.tensor([0.1], device=device, dtype=dtype))

theta_max   = 0.35
track_limit = 1.0

# ================================================================
# 4. GPU dynamics
# ================================================================
def step_gpu(fn_gpu, x, u, w, params):
    fn_gpu.evaluate([x, u, w, *params])
    return fn_gpu.outputs_sparse[0].clone()

# ================================================================
# 5. CVaR evaluation of candidate sequences
# ================================================================
# def evaluate_candidates_cvar(x0, U_cand):
#     """
#     x0:     (NX,)
#     U_cand: (K,N,NU)
#     returns: total_cost(K,), mean_cost(K,), cvar(K,)
#     """
#     K, N, nu = U_cand.shape
#     S = NUM_SCENARIOS
#     B = K * S

#     # repeat initial state
#     x = x0.view(1,NX).repeat(B,1)

#     traj_cost = torch.zeros(B, device=device, dtype=dtype)

#     for t in range(N):
#         # controls
#         U_t = U_cand[:,t,:].unsqueeze(1).repeat(1,S,1)
#         u = U_t.view(B,NU)

#         # disturbances
#         w = WIND_STD * torch.randn((B,2), device=device, dtype=dtype)

#         # propagate
#         x = step_gpu(fn_gpu_saa, x, u, w, params_saa)

#         # quadratic cost
#         dx = x - x_trim
#         du = u - u_trim
#         quad = (dx @ Q * dx).sum(dim=1) + (du @ R * du).sum(dim=1)

#         # soft chance penalties
#         th = x[:,2]
#         p  = x[:,0]

#         tilt_violation  = F.softplus(10.0 * (torch.abs(th) - theta_max)) / 10.0
#         track_violation = F.softplus(10.0 * (torch.abs(p) - track_limit)) / 10.0

#         stage_cost = quad + LAMBDA_PEN * (tilt_violation + track_violation)
#         traj_cost += stage_cost

#     # reshape to (K,S)
#     traj_cost = traj_cost.view(K,S)

#     # mean cost
#     mean_cost = traj_cost.mean(dim=1)

#     # CVaR cost (worst α fraction)
#     S_tail = max(1, int(np.ceil(ALPHA_CVAR * S)))
#     sorted_cost,_ = torch.sort(traj_cost, dim=1, descending=True)
#     tail_cost = sorted_cost[:, :S_tail]
#     cvar = tail_cost.mean(dim=1)

#     # CVaR objective
#     total_cost = mean_cost + LAMBDA_RISK * (cvar - mean_cost)
#     return total_cost, mean_cost, cvar

def evaluate_candidates_cvar(x0, U_cand):
    K, N, nu = U_cand.shape
    S = NUM_SCENARIOS
    B = K * S

    # repeat initial state
    x = x0.view(1,NX).repeat(B,1)

    traj_cost = torch.zeros(B, device=device, dtype=dtype)

    for t in range(N):
        # controls
        U_t = U_cand[:,t,:].unsqueeze(1).repeat(1,S,1)
        u = U_t.view(B,NU)

        # disturbances
        w = WIND_STD * torch.randn((B,2), device=device, dtype=dtype)

        # propagate
        x = step_gpu(fn_gpu_saa, x, u, w, params_saa)

        # quadratic
        dx = x - x_trim
        du = u - u_trim
        quad = (dx @ Q * dx).sum(dim=1) + (du @ R * du).sum(dim=1)

        # soft chance penalties
        th = x[:,2]
        p  = x[:,0]

        tilt_violation  = F.softplus(5.0*(torch.abs(th) - theta_max))/5.0
        track_violation = F.softplus(5.0*(torch.abs(p)  - track_limit))/5.0

        stage_cost = quad + LAMBDA_PEN*(tilt_violation + track_violation)
        traj_cost += stage_cost

    traj_cost = traj_cost.view(K,S)

    # mean loss
    mean_cost = traj_cost.mean(dim=1)

    # correct CVaR
    VaR = traj_cost.quantile(1 - ALPHA_CVAR, dim=1, interpolation="linear")

    excess = torch.clamp(traj_cost - VaR.unsqueeze(1), min=0.0)
    CVaR  = VaR + (1.0/ALPHA_CVAR)*excess.mean(dim=1)

    # combined risk objective
    total_cost = (1 - LAMBDA_RISK)*mean_cost + LAMBDA_RISK*CVaR

    return total_cost, mean_cost, CVaR


# ================================================================
# 6. MPC: random shooting
# ================================================================
def sample_candidates(mean_U, std):
    noise = std * torch.randn((NUM_CANDIDATES,HORIZON,NU), device=device, dtype=dtype)
    return mean_U.view(1,HORIZON,NU) + noise

# ================================================================
# 7. Rollout
# ================================================================
x_eval = torch.zeros((EVAL_BATCH, NX), device=device, dtype=dtype)
x_eval[:,2] = 0.45
x_eval[:,0] = 1.2

theta_violation_curve = []
track_violation_curve = []

px_log    = torch.zeros((STEPS,PLOTS), device=device)
theta_log = torch.zeros((STEPS,PLOTS), device=device)

mean_U = u_trim.view(1,NU).repeat(HORIZON,1)
std_U = 0.5

print("Running CVaR MPC rollout...")

for t in range(STEPS):

    # 1) sample U sequences
    U_cand = sample_candidates(mean_U, std_U)

    # 2) CVaR evaluation
    costs, mean_costs, cvar_costs = evaluate_candidates_cvar(x_eval[0], U_cand)

    # 3) choose best
    k_best = torch.argmin(costs).item()
    U_best = U_cand[k_best]
    u_t = U_best[0]

    # warm start
    mean_U = torch.roll(U_best, shifts=-1, dims=0)
    mean_U[-1] = u_trim

    # 4) apply to evaluation batch
    u_batch = u_t.view(1,NU).repeat(EVAL_BATCH,1)
    w_eval = WIND_STD * torch.randn((EVAL_BATCH,2), device=device, dtype=dtype)

    x_eval = step_gpu(fn_gpu_eval, x_eval, u_batch, w_eval, params_eval)

    # stats
    theta_violation_curve.append((torch.abs(x_eval[:,2]) > theta_max).double().mean().item())
    track_violation_curve.append((torch.abs(x_eval[:,0]) > track_limit).double().mean().item())

    px_log[t]    = x_eval[:PLOTS,0]
    theta_log[t] = x_eval[:PLOTS,2]

torch.cuda.synchronize()
print("CVaR MPC rollout complete.")

# ================================================================
# 8. Results + Plots
# ================================================================
print("\n=== CVaR MPC Summary ===")
print("Final pole violation prob:", theta_violation_curve[-1])
print("Final track violation prob:", track_violation_curve[-1])

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(14,5))
plt.plot(theta_violation_curve, label="Pole violation")
plt.plot(track_violation_curve, label="Track violation")
plt.axhline(0.05, color="red", linestyle="--")
plt.title("CVaR MPC Chance Violations")
plt.grid(True)
plt.legend()
plt.savefig("results/cartpole_cvar_chance.png")
plt.show()

px = px_log.cpu().numpy()
th = theta_log.cpu().numpy()

fig,ax = plt.subplots(1,2, figsize=(13,5))
for i in range(PLOTS):
    ax[0].plot(px[:,i])
    ax[1].plot(th[:,i])

ax[0].set_title("Cart Position — CVaR MPC")
ax[1].set_title("Pole Angle — CVaR MPC")
plt.savefig("results/cartpole_cvar_trajs.png")
plt.show()

print("Saved CVaR MPC results.")
