import os
import torch
from casadi import *
from src import *  # should provide CusadiFunction etc.

"""
Sanity check for fn_quad_sim_step:
 - BATCH_SIZE = 1
 - runs on CPU
 - prints state evolution for a single quadrotor
"""

# === 1. Settings ===
BATCH_SIZE = 1
NUM_STEPS = 10
CUSADI_FUNCTION_DIR = "src/casadi_functions"

device = "cuda"
dtype = torch.double

# === 2. Load CasADi + CusADi function ===
fn_casadi_quad_step = casadi.Function.load(
    os.path.join(CUSADI_FUNCTION_DIR, "fn_quad_sim_step.casadi")
)
print("Loaded CasADi function:", fn_casadi_quad_step)

# Wrap with CusADi
fn_cusadi_quad_step = CusadiFunction(fn_casadi_quad_step, BATCH_SIZE)
print("CusADi quadrotor function created with batch size:", BATCH_SIZE)

# === 3. Define initial state and parameters ===
# State: [p_x, p_z, theta, v_x, v_z, omega]
# Give it some tilt and horizontal velocity so it actually moves.
x = torch.tensor([[0.0, 1.0, 0.2, 0.5, 0.0, 0.0]], dtype=dtype, device=device)

# Physical parameters (scalars, but batched as [B, 1] tensors for CusADi)
m_val = 1.0
I_val = 0.05
l_val = 0.2
g_val = 9.81
dt_val = 0.01

m = m_val * torch.ones((BATCH_SIZE, 1), dtype=dtype, device=device)
I = I_val * torch.ones((BATCH_SIZE, 1), dtype=dtype, device=device)
l = l_val * torch.ones((BATCH_SIZE, 1), dtype=dtype, device=device)
g = g_val * torch.ones((BATCH_SIZE, 1), dtype=dtype, device=device)
dt = dt_val * torch.ones((BATCH_SIZE, 1), dtype=dtype, device=device)

# Control: hover thrust (T1 = T2 = mg/2)
T_hover = m_val * g_val / 2.0
u = torch.zeros((BATCH_SIZE, 2), dtype=dtype, device=device)
u[:, 0] = T_hover
u[:, 1] = T_hover

# Disturbance parameters
wind_std = 0.5  # make it big enough to see an effect

print("\nInitial state x:", x)
print("Hover thrust per rotor:", T_hover)
print()

# === 4. Step the dynamics a few times and print ===
for step in range(NUM_STEPS):
    # Sample new wind each step: w = [w_x, w_z]
    w = wind_std * torch.randn((BATCH_SIZE, 2), dtype=dtype, device=device)

    print(f"Step {step}:")
    print("  x before =", x)

    # Call CusADi quadrotor step: fn_quad_sim_step(x, u, w, m, I, l, g, dt)
    fn_cusadi_quad_step.evaluate([x, u, w, m, I, l, g, dt])
    x = fn_cusadi_quad_step.outputs_sparse[0].clone()

    print("  w        =", w)
    print("  x after  =", x)
    print()

print("Sanity-check run complete.")
