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

HORIZON = 30              # time horizon
NUM_CANDIDATES = 128      # K candidate control sequences
NUM_SCENARIOS  = 16       # S disturbance scenarios per candidate

EVAL_BATCH = 4096          # evaluation batch size
STEPS = 500
PLOTS = 10

WIND_STD = 1.5             # disturbance level

ALPHA = 0.05               # used for reference line in plots
lambda_penalty = 2000.0    # soft-penalty weight


print("\n=== SAA MPC (Chance-Constrained) — CartPole ===")

# ================================================================
# 2. Load CasADi model + GPU wrappers
# ================================================================
fn_casadi = ca.Function.load("src/casadi_functions/fn_cartpole_sim_step.casadi")

# Batch for SAA: K * S
BATCH_SAA = NUM_CANDIDATES * NUM_SCENARIOS
fn_gpu_saa = CusadiFunction(fn_casadi, BATCH_SAA)

# Evaluation batch
fn_gpu_eval = CusadiFunction(fn_casadi, EVAL_BATCH)

# physical parameters
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
# 3. Trim + Cost weights + Constraints
# ================================================================
x_trim = torch.tensor([0.,0.,0.,0.], device=device, dtype=dtype)
u_trim = torch.tensor([0.], device=device, dtype=dtype)

Q = torch.diag(torch.tensor([10., 1., 40., 5.], device=device, dtype=dtype))
R = torch.diag(torch.tensor([0.1], device=device, dtype=dtype))

theta_max = 0.35
track_limit = 1.0     # absolute cart position limit

print("Trim:", x_trim)
print("u_trim:", u_trim)

# ================================================================
# 4. GPU dynamics wrapper
# ================================================================
def step_gpu(fn_gpu, x, u, w, params):
    fn_gpu.evaluate([x, u, w, *params])
    return fn_gpu.outputs_sparse[0].clone()

# ================================================================
# 5. SAA evaluation of candidate sequences
# ================================================================
def evaluate_candidates_saa(x0, U_cand):
    """
    x0: (NX,)
    U_cand: (K, N, NU)
    returns: mean_cost (K,)
    """
    K, N, nu = U_cand.shape
    S = NUM_SCENARIOS
    B = K * S

    x = x0.view(1, NX).repeat(B, 1)   # (B, NX)
    cost_env = torch.zeros(B, device=device, dtype=dtype)

    for t in range(N):
        # controls: (K,1,NU) -> (K,S,NU) -> (B,NU)
        U_t = U_cand[:, t, :].unsqueeze(1).repeat(1, S, 1)
        u = U_t.view(B, nu)

        # disturbances
        w = WIND_STD * torch.randn((B,2), device=device, dtype=dtype)

        # step dynamics
        x = step_gpu(fn_gpu_saa, x, u, w, params_saa)

        # quadratic cost
        dx = x - x_trim.view(1, NX)
        du = u - u_trim.view(1, NU)
        quad = (dx @ Q * dx).sum(dim=1) + (du @ R * du).sum(dim=1)

        # soft constraints
        th = x[:,2]
        p  = x[:,0]

        tilt_violation = F.softplus(10.0 * (torch.abs(th) - theta_max)) / 10.0
        track_violation = F.softplus(10.0 * (torch.abs(p) - track_limit)) / 10.0

        stage_cost = quad + lambda_penalty * (tilt_violation + track_violation)
        cost_env += stage_cost

    # reshape -> average over scenarios
    cost_env = cost_env.view(K, S)
    mean_cost = cost_env.mean(dim=1)
    return mean_cost


# ================================================================
# 6. MPC loop (random shooting)
# ================================================================
def sample_candidates(mean_U, std):
    K = NUM_CANDIDATES
    noise = std * torch.randn((K, HORIZON, NU), device=device, dtype=dtype)
    return mean_U.view(1,HORIZON,NU) + noise


# initial evaluation rollout state
x_eval = torch.zeros((EVAL_BATCH, NX), dtype=dtype, device=device)
x_eval[:,2] = 0.45    # challenging initial pole angle
x_eval[:,0] = 1.2

theta_violation_curve = []
track_violation_curve = []

px_log = torch.zeros((STEPS, PLOTS), device=device)
theta_log = torch.zeros((STEPS, PLOTS), device=device)

# warm-start control guess
mean_U = u_trim.view(1,NU).repeat(HORIZON,1)
std_U = 0.5

print("Running SAA MPC rollout...")

for t in range(STEPS):
    # 1) sample candidate sequences
    U_cand = sample_candidates(mean_U, std_U)

    # 2) evaluate SAA cost
    x0_single = x_eval[0].clone()
    mean_cost = evaluate_candidates_saa(x0_single, U_cand)

    # 3) pick best candidate
    k_best = torch.argmin(mean_cost).item()
    U_best = U_cand[k_best]
    u_t = U_best[0]

    # 4) warm-start next iteration
    mean_U = torch.roll(U_best, shifts=-1, dims=0)
    mean_U[-1] = u_trim

    # 5) evaluate on large batch (fresh noise)
    u_batch = u_t.view(1,NU).repeat(EVAL_BATCH,1)
    w_eval = WIND_STD * torch.randn((EVAL_BATCH,2), dtype=dtype, device=device)

    x_eval = step_gpu(fn_gpu_eval, x_eval, u_batch, w_eval, params_eval)

    # 6) log violations
    theta_violation_curve.append((torch.abs(x_eval[:,2]) > theta_max).double().mean().item())
    track_violation_curve.append((torch.abs(x_eval[:,0]) > track_limit).double().mean().item())

    px_log[t] = x_eval[:PLOTS,0]
    theta_log[t] = x_eval[:PLOTS,2]

torch.cuda.synchronize()
print("SAA MPC completed.")

# ================================================================
# 7. Results summary
# ================================================================
print("\n=== SAA MPC Summary ===")
print("Final pole-angle violation prob:", theta_violation_curve[-1])
print("Final track violation prob:", track_violation_curve[-1])

# ================================================================
# 8. Plots
# ================================================================
os.makedirs("results", exist_ok=True)

# chance constraints
plt.figure(figsize=(14,5))
plt.plot(theta_violation_curve, label="Pole violation prob")
plt.plot(track_violation_curve, label="Track violation prob")
plt.axhline(ALPHA, color="red", linestyle="--", label="α = 0.05")
plt.title("Chance Violations — SAA MPC")
plt.xlabel("Time")
plt.ylabel("Probability")
plt.grid(True)
plt.legend()
plt.savefig("results/cartpole_saa_chance.png")
plt.show()

# trajectories
px = px_log.cpu().numpy()
th = theta_log.cpu().numpy()

fig, ax = plt.subplots(1,2, figsize=(13,5))
for i in range(PLOTS):
    ax[0].plot(px[:,i])
    ax[1].plot(th[:,i])

ax[0].set_title("Cart Position p — SAA MPC")
ax[1].set_title("Pole Angle θ — SAA MPC")
plt.savefig("results/cartpole_saa_trajs.png")
plt.show()

print("Saved SAA MPC results.")
