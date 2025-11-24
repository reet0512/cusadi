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

NX = 6
NU = 2

HORIZON = 30        # same as SAA
WIND_STD = 0.2
EVAL_BATCH = 4096
STEPS = 500
PLOTS = 10

# ============================================================
# 2. Load model
# ============================================================
CUSADI_FUNCTION_DIR = "src/casadi_functions"
fn_casadi = ca.Function.load(os.path.join(CUSADI_FUNCTION_DIR,
                                          "fn_quad_sim_step.casadi"))

fn_gpu_eval = CusadiFunction(fn_casadi, EVAL_BATCH)

# physical params
m_val  = 1.0
I_val  = 0.05
l_val  = 0.2
g_val  = 9.81
dt_val = 0.01

def make_params(batch):
    return [
        torch.full((batch,1), m_val,  dtype=dtype, device=device),
        torch.full((batch,1), I_val,  dtype=dtype, device=device),
        torch.full((batch,1), l_val,  dtype=dtype, device=device),
        torch.full((batch,1), g_val,  dtype=dtype, device=device),
        torch.full((batch,1), dt_val, dtype=dtype, device=device)
    ]

params_eval = make_params(EVAL_BATCH)

# ============================================================
# 3. Trim + cost weights
# ============================================================
T_hover = m_val * g_val
x_trim_np = np.array([0., 1., 0., 0., 0., 0.])
u_trim_np = np.array([T_hover/2, T_hover/2])

x_trim = torch.tensor(x_trim_np, device=device, dtype=dtype)
u_trim = torch.tensor(u_trim_np, device=device, dtype=dtype)

Q = torch.diag(torch.tensor([40.,40.,10.,2.,2.,1.], device=device, dtype=dtype))
R = torch.diag(torch.tensor([0.4,0.4], device=device, dtype=dtype))

print("=== Deterministic Shooting MPC (Matching SAA Style) ===")
print("Trim:", x_trim_np, u_trim_np)

# ============================================================
# 4. GPU step wrapper
# ============================================================
def step_gpu(fn_gpu, x, u, w, params):
    fn_gpu.evaluate([x, u, w, *params])
    return fn_gpu.outputs_sparse[0].clone()

# ============================================================
# 5. Evaluate a SINGLE candidate sequence (deterministic)
# ============================================================
def eval_det_candidate(x0, U_seq):
    """
    U_seq: (N,NU)
    returns scalar deterministic cost (no SAA)
    """
    x = x0.clone()
    cost = 0.0

    for t in range(HORIZON):
        u = U_seq[t]

        # no disturbance
        w = torch.zeros((1,2), device=device, dtype=dtype)

        x = step_gpu(fn_gpu_eval, x.view(1,NX), 
                     u.view(1,NU), w, params_eval)[0]

        dx = x - x_trim
        du = u - u_trim

        stage_cost = (dx @ Q @ dx) + (du @ R @ du)
        cost += stage_cost

    return cost.item()

# ============================================================
# 6. Deterministic MPC Policy = gradient-free local improvement
# ============================================================
def mpc_det_policy(x0, mean_U, std):
    """
    One-step deterministic shooting MPC:
      - small Gaussian perturbation around mean_U
      - pick candidate with lowest deterministic cost
    """

    K = 128   # much smaller than SAA (128)
    U_cand = mean_U + std * torch.randn((K,HORIZON,NU), device=device, dtype=dtype)

    costs = []
    for k in range(K):
        c = eval_det_candidate(x0, U_cand[k])
        costs.append(c)

    best = int(np.argmin(costs))
    return U_cand[best]

# ============================================================
# 7. Evaluate full MPC performance over 4096 environments
# ============================================================
x_eval = torch.zeros((EVAL_BATCH, NX), dtype=dtype, device=device)
x_eval[:,1] = 1.0

tilt_violation_curve = []
alt_violation_curve = []

px_log = torch.zeros((STEPS, PLOTS), device=device)
pz_log = torch.zeros((STEPS, PLOTS), device=device)
theta_log = torch.zeros((STEPS, PLOTS), device=device)

# initial warm-start control sequence
mean_U = u_trim.view(1,NU).repeat(HORIZON,1)
std_U  = 1.0

for t in range(STEPS):
    # 1) choose deterministic best sequence
    best_seq = mpc_det_policy(x_eval[0], mean_U, std_U)
    u_t = best_seq[0]

    # 2) warm start next iteration
    mean_U = torch.roll(best_seq, shifts=-1, dims=0)
    mean_U[-1] = u_trim

    # 3) apply control to evaluation batch
    u_batch = u_t.view(1,NU).repeat(EVAL_BATCH,1)
    w = WIND_STD * torch.randn((EVAL_BATCH,2), device=device, dtype=dtype)

    x_eval = step_gpu(fn_gpu_eval, x_eval, u_batch, w, params_eval)

    # 4) log violations
    tilt_violation_curve.append((torch.abs(x_eval[:,2]) > 0.35).double().mean().item())
    alt_violation_curve.append((x_eval[:,1] < 1.0).double().mean().item())

    px_log[t] = x_eval[:PLOTS,0]
    pz_log[t] = x_eval[:PLOTS,1]
    theta_log[t] = x_eval[:PLOTS,2]

torch.cuda.synchronize()
print("Deterministic shooting MPC done.")

print("\n=== Summary ===")
print("Final tilt prob:", tilt_violation_curve[-1])
print("Final alt prob:", alt_violation_curve[-1])

# ============================================================
# 8. Plots
# ============================================================
plt.figure(figsize=(14,5))
plt.plot(tilt_violation_curve, label="Tilt violation prob")
plt.plot(alt_violation_curve, label="Alt violation prob")
plt.axhline(0.05, color="red", linestyle="--")
plt.grid(True)
plt.legend()
plt.title("Deterministic Shooting MPC — Chance Evaluation")
plt.savefig("results/mpc_det_shooting_chance.png")
plt.show()

px = px_log.cpu().numpy()
pz = pz_log.cpu().numpy()

fig,ax = plt.subplots(1,2,figsize=(13,5))
for i in range(PLOTS):
    ax[0].plot(px[:,i])
    ax[1].plot(pz[:,i])

ax[0].set_title("px trajectories")
ax[1].set_title("pz trajectories")
plt.savefig("results/mpc_det_shooting_trajs.png")
plt.show()

print("Saved deterministic shooting MPC results.")
