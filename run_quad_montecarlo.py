# run_quad_montecarlo.py
import os
import torch
import matplotlib.pyplot as plt
from casadi import *
from src import *

# ======================================
# Settings
# ======================================
CUSADI_FUNCTION_DIR = "src/casadi_functions"
N = 10000          # Monte Carlo samples
T = 300            # horizon steps
DT = 0.02          # timestep

# Disturbance statistics (PDF only uses w_x, w_z)
W_MEAN = torch.tensor([0.0, 0.0], dtype=torch.double)
W_STD  = torch.tensor([1.0, 1.0], dtype=torch.double)

os.makedirs("results", exist_ok=True)

# ======================================
# 1. Load compiled quadrotor function
# ======================================
f = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "fn_quad_step.casadi"))
fn = CusadiFunction(f, N)

print("Loaded fn_quad_step ✓")

# ======================================
# 2. Initial state batch (GPU)
# ======================================
x = torch.zeros((N, 6), device='cuda', dtype=torch.double)
x[:,2] = 0.05 * torch.randn(N, device='cuda')   # v_x perturbation
x[:,4] = 0.02 * torch.randn(N, device='cuda')   # small tilt perturbation

# ======================================
# 3. Hover control input
# (T1 + T2 = mg → 9.81 => 4.9 + 4.9)
# ======================================
u = torch.tensor([[4.9, 4.9]], device='cuda', dtype=torch.double).repeat(N, 1)

# ======================================
# 4. Parameters
# ======================================
m  = 1.0  * torch.ones((N,1), device='cuda', dtype=torch.double)
I  = 0.02 * torch.ones((N,1), device='cuda', dtype=torch.double)
l  = 0.25 * torch.ones((N,1), device='cuda', dtype=torch.double)
g  = 9.81 * torch.ones((N,1), device='cuda', dtype=torch.double)
dt = DT   * torch.ones((N,1), device='cuda', dtype=torch.double)

# ======================================
# 5. Logging
# ======================================
px_log = torch.zeros((T, N), device='cuda')
vx_log = torch.zeros((T, N), device='cuda')

# ======================================
# 6. Monte-Carlo rollout
# ======================================
print("Running Monte-Carlo rollout...")

for t in range(T):

    # Disturbance batch: (N, 2)
    w = (W_MEAN.to('cuda')
         + W_STD.to('cuda') * torch.randn((N, 2), device='cuda', dtype=torch.double))

    fn.evaluate([x, u, w, m, I, l, g, dt])
    x = fn.outputs_sparse[0].clone()       # IMPORTANT clone-fix

    px_log[t] = x[:,0]
    vx_log[t] = x[:,2]

torch.cuda.synchronize()
print("Monte-Carlo rollout complete ✓")

# ======================================
# 7. Statistics
# ======================================
mean_px = px_log.mean(dim=1).cpu()
std_px  = px_log.std(dim=1).cpu()

mean_vx = vx_log.mean(dim=1).cpu()
std_vx  = vx_log.std(dim=1).cpu()

print("\nNUMERICAL CHECKS:")
print("max |mean_px| =", float(mean_px.abs().max()))
print("max  std_px  =", float(std_px.max()))
print("max |mean_vx| =", float(mean_vx.abs().max()))
print("max  std_vx  =", float(std_vx.max()))

# Time axis
t_axis = DT * torch.arange(T)

# ======================================
# 8. Plot p_x distribution
# ======================================
plt.figure(figsize=(10,4))
plt.plot(t_axis, mean_px, label="E[p_x(t)]")
plt.fill_between(t_axis, mean_px-std_px, mean_px+std_px, alpha=0.3)
plt.grid(True)
plt.xlabel("time (s)")
plt.ylabel("p_x (m)")
plt.title("Monte-Carlo distribution of p_x (PDF model)")
plt.legend()
plt.tight_layout()
plt.savefig("results/montecarlo_px.png", dpi=150)
plt.show()

# ======================================
# 9. Plot v_x distribution
# ======================================
plt.figure(figsize=(10,4))
plt.plot(t_axis, mean_vx, label="E[v_x(t)]")
plt.fill_between(t_axis, mean_vx-std_vx, mean_vx+std_vx, alpha=0.3)
plt.grid(True)
plt.xlabel("time (s)")
plt.ylabel("v_x (m/s)")
plt.title("Monte-Carlo distribution of v_x (PDF model)")
plt.legend()
plt.tight_layout()
plt.savefig("results/montecarlo_vx.png", dpi=150)
plt.show()

print("\n✓ All plots saved. Monte-Carlo for PDF model complete.")
