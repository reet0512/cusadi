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

NX = 6
NU = 2

# MPC + SAA hyperparameters
HORIZON = 30          # N
NUM_CANDIDATES = 128  # K (control sequences)
NUM_SCENARIOS = 16    # S (disturbance samples per candidate)
WIND_STD = 0.2
ALPHA = 0.05          # for reference in plots

# Outer evaluation batch (for chance curves)
EVAL_BATCH = 4096
STEPS = 500
PLOTS = 10

# ==========================
# 2. Load CasADi quadrotor step and wrap with CusADi
# ==========================
CUSADI_FUNCTION_DIR = "src/casadi_functions"
fn_casadi = ca.Function.load(os.path.join(CUSADI_FUNCTION_DIR,
                                          "fn_quad_sim_step.casadi"))

# For SAA optimization: batch = NUM_CANDIDATES * NUM_SCENARIOS
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

# Torch versions of parameters
def make_params(batch, device=device, dtype=dtype):
    m  = torch.full((batch,1), m_val,  dtype=dtype, device=device)
    I  = torch.full((batch,1), I_val,  dtype=dtype, device=device)
    l  = torch.full((batch,1), l_val,  dtype=dtype, device=device)
    g  = torch.full((batch,1), g_val,  dtype=dtype, device=device)
    dt = torch.full((batch,1), dt_val, dtype=dtype, device=device)
    return [m, I, l, g, dt]

params_saa  = make_params(BATCH_SAA)
params_eval = make_params(EVAL_BATCH)

# ==========================
# 3. Trim point and cost weights
# ==========================
T_hover = m_val * g_val
x_trim_np = np.array([0., 1., 0., 0., 0., 0.])
u_trim_np = np.array([T_hover/2, T_hover/2])

x_trim = torch.tensor(x_trim_np, device=device, dtype=dtype)
u_trim = torch.tensor(u_trim_np, device=device, dtype=dtype)

Q = torch.diag(torch.tensor([40., 40., 10., 2., 2., 1.], device=device, dtype=dtype))
R = torch.diag(torch.tensor([0.4, 0.4], device=device, dtype=dtype))

theta_max = 0.35
z_min     = 1.0
lambda_penalty = 2000.0   # weight on soft chance penalties

print("=== GPU SAA MPC ===")
print("x_trim:", x_trim_np)
print("u_trim:", u_trim_np)


# ==========================
# 4. Helper: one-step GPU dynamics via CusADi
# ==========================
def step_gpu(fn_gpu, x, u, w, params):
    """
    x: (B, NX), u: (B, NU), w: (B, 2)
    params: list of (B,1) tensors [m, I, l, g, dt]
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

    # Prepare tensor to accumulate per-env costs
    cost_env = torch.zeros(B, device=device, dtype=dtype)

    # Pre-make params (already created globally)
    params = params_saa

    # Convenience view
    # For each k: U_k (K, NU) → repeat over S: (K, S, NU) → (B, NU)
    for t in range(N):
        U_t = U_cand[:, t, :]                             # (K, NU)
        U_t = U_t.unsqueeze(1).repeat(1, S, 1)           # (K, S, NU)
        u = U_t.view(B, nu)                               # (B, NU)

        # Sample disturbances for all envs at this time
        w = WIND_STD * torch.randn((B, 2), device=device, dtype=dtype)

        # One-step dynamics
        x = step_gpu(fn_gpu_saa, x, u, w, params)

        # State and control deviations
        dx = x - x_trim.view(1, NX)
        du = u - u_trim.view(1, NU)

        # Quadratic LQR-like cost
        quad = (dx @ Q * dx).sum(dim=1) + (du @ R * du).sum(dim=1)

        # Soft chance penalties
        theta = x[:, 2]
        z     = x[:, 1]
        tilt_violation = F.softplus(10.0 * (torch.abs(theta) - theta_max)) / 10.0
        alt_violation  = F.softplus(10.0 * (z_min - z)) / 10.0

        stage_cost = quad + lambda_penalty * (tilt_violation + alt_violation)
        cost_env += stage_cost

    # Reshape (K, S), average over scenarios → mean SAA cost for each candidate
    cost_env = cost_env.view(K, S)
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
x_eval[:, 1] = 1.0

tilt_violation_curve = []
alt_violation_curve  = []

px_log = torch.zeros((STEPS, PLOTS), device=device)
pz_log = torch.zeros((STEPS, PLOTS), device=device)

# Initial mean control sequence: hover thrust
mean_U = u_trim.view(1, NU).repeat(HORIZON, 1)  # (N, NU)
std_U  = 2.0  # initial sampling std (N)

for t in range(STEPS):
    # 1) Build candidate sequences around mean_U
    U_cand = sample_candidates(mean_U, std_U)   # (K, N, NU)

    # 2) Evaluate SAA costs on GPU
    x0_single = x_eval[0].clone()              # use env 0 as nominal
    mean_cost = evaluate_candidates_saa(x0_single, U_cand)

    # 3) Pick best candidate
    k_best = torch.argmin(mean_cost).item()
    U_best = U_cand[k_best]                    # (N, NU)
    u_t    = U_best[0]                         # first control in the sequence

    # Optional: update mean_U for next step (receding-horizon warm-start)
    mean_U = torch.roll(U_best, shifts=-1, dims=0)
    mean_U[-1] = u_trim  # terminal guess

    # 4) Apply chosen u_t to *evaluation* batch under fresh noise
    u_eval = u_t.view(1, NU).repeat(EVAL_BATCH, 1)
    w_eval = WIND_STD * torch.randn((EVAL_BATCH, 2), dtype=dtype, device=device)

    x_eval = step_gpu(fn_gpu_eval, x_eval, u_eval, w_eval, params_eval)

    # 5) Log chance violations
    tilt_violation_curve.append((torch.abs(x_eval[:,2]) > theta_max).double().mean().item())
    alt_violation_curve.append((x_eval[:,1] < z_min).double().mean().item())

    # Log some trajectories
    px_log[t] = x_eval[:PLOTS, 0]
    pz_log[t] = x_eval[:PLOTS, 1]

torch.cuda.synchronize()
print("SAA MPC simulation complete!")

print("\n================ GPU SAA MPC Summary ================")
print(f"Final tilt violation prob = {tilt_violation_curve[-1]:.4f}")
print(f"Final alt violation prob  = {alt_violation_curve[-1]:.4f}")
print("=====================================================")


# ==========================
# 7. Plot chance violations
# ==========================
plt.figure(figsize=(14,5))
plt.plot(tilt_violation_curve, label="Tilt violation probability")
plt.plot(alt_violation_curve, label="Altitude violation probability")
plt.axhline(ALPHA, color="red", linestyle="--", label=f"α = {ALPHA}")
plt.xlabel("Time Step")
plt.ylabel("Probability")
plt.title(f"Chance Constraint Violations Over Time — GPU SAA MPC "
          f"(K={NUM_CANDIDATES}, S={NUM_SCENARIOS})")
plt.grid(True)
plt.legend()
os.makedirs("results", exist_ok=True)
plt.savefig("results/mpc_chance_saa_gpu.png", dpi=150)
plt.show()

# ==========================
# 8. Plot trajectories
# ==========================
px = px_log.cpu().numpy()
pz = pz_log.cpu().numpy()

fig, ax = plt.subplots(1,2, figsize=(13,5))
for i in range(PLOTS):
    ax[0].plot(px[:, i])
    ax[1].plot(pz[:, i])

ax[0].set_title("Horizontal Position — GPU SAA MPC")
ax[0].set_xlabel("Time Step")
ax[0].set_ylabel("px (m)")
ax[0].grid(True)

ax[1].set_title("Vertical Position — GPU SAA MPC")
ax[1].set_xlabel("Time Step")
ax[1].set_ylabel("pz (m)")
ax[1].grid(True)

plt.tight_layout()
plt.savefig("results/mpc_chance_saa_gpu_trajs.png", dpi=150)
plt.show()
print("Saved plots to results/ directory.")
