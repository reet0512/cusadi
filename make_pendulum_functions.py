import casadi as ca
import os

# === 1. Define symbolic variables ===
x_pend = ca.SX.sym('x_pend', 2, 1)   # [theta, omega]
g = ca.SX.sym('g')                   # gravity
l = ca.SX.sym('l')                   # length
dt = ca.SX.sym('dt')                 # timestep

# === 2. Pendulum dynamics ===
f_pend = ca.vertcat(
    x_pend[1],           # theta_dot = omega
    -g * ca.sin(x_pend[0]) / l   # omega_dot
)

# === 3. Jacobian ===
J_pend = ca.jacobian(f_pend, x_pend)

# === 4. Semi-implicit Euler integration step ===
omega_next = x_pend[1] - (g * ca.sin(x_pend[0]) / l) * dt
theta_next = x_pend[0] + omega_next * dt
x_next_pend = ca.vertcat(theta_next, omega_next)

# === 5. Create CasADi Functions ===
fn_dynamics = ca.Function('fn_dynamics', [x_pend, g, l], [f_pend])
fn_jacobian = ca.Function('fn_jacobian', [x_pend, g, l], [J_pend])
fn_sim_step = ca.Function('fn_sim_step', [x_pend, g, l, dt], [x_next_pend])

# === 6. Save them ===
out_dir = os.path.join('src', 'casadi_functions')
os.makedirs(out_dir, exist_ok=True)

fn_dynamics.save(os.path.join(out_dir, 'fn_dynamics.casadi'))
fn_jacobian.save(os.path.join(out_dir, 'fn_jacobian.casadi'))
fn_sim_step.save(os.path.join(out_dir, 'fn_sim_step.casadi'))

print("✅ CasADi functions saved to", out_dir)
