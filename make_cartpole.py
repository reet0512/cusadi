import casadi as ca
import os

# ============================================================
# 1. Define symbolic variables
# ============================================================

# State: [p, p_dot, theta, theta_dot]
x = ca.SX.sym('x', 4, 1)

# Control: horizontal force F
u = ca.SX.sym('u', 1, 1)

# Disturbance: [w_p, w_theta]
w = ca.SX.sym('w', 2, 1)

# Physical parameters
m_c = ca.SX.sym('m_c')   # cart mass
m_p = ca.SX.sym('m_p')   # pole mass
l   = ca.SX.sym('l')     # half-length of pole
g   = ca.SX.sym('g')     # gravity
dt  = ca.SX.sym('dt')    # timestep

# ============================================================
# 2. Unpack states & inputs
# ============================================================

p, p_dot, theta, theta_dot = x[0], x[1], x[2], x[3]
F = u[0]
w_p, w_theta = w[0], w[1]

M = m_c + m_p

# ============================================================
# 3. Continuous-time dynamics
# ============================================================

# denominator used for angular acceleration
den = l * (4.0/3.0 - (m_p * ca.cos(theta)**2) / M)

# angular acceleration
theta_ddot = (
    g * ca.sin(theta)
    + ca.cos(theta) * (-F - m_p * l * theta_dot**2 * ca.sin(theta)) / M
) / den + w_theta

# cart acceleration
p_ddot = (F + m_p * l * (theta_dot**2 * ca.sin(theta) - theta_ddot * ca.cos(theta))) / M + w_p

# continuous dynamics vector
f_cont = ca.vertcat(
    p_dot,
    p_ddot,
    theta_dot,
    theta_ddot
)

# ============================================================
# 4. Jacobians
# ============================================================

J_x = ca.jacobian(f_cont, x)
J_u = ca.jacobian(f_cont, u)

# ============================================================
# 5. Semi-implicit Euler discretization
# ============================================================

# update velocities
p_dot_next     = p_dot     + p_ddot     * dt
theta_dot_next = theta_dot + theta_ddot * dt

# update positions
p_next     = p     + p_dot_next     * dt
theta_next = theta + theta_dot_next * dt

x_next = ca.vertcat(
    p_next,
    p_dot_next,
    theta_next,
    theta_dot_next
)

# ============================================================
# 6. Build CasADi functions
# ============================================================

fn_cartpole_dynamics = ca.Function(
    'fn_cartpole_dynamics',
    [x, u, w, m_c, m_p, l, g],
    [f_cont]
)

fn_cartpole_jacobian_x = ca.Function(
    'fn_cartpole_jacobian_x',
    [x, u, w, m_c, m_p, l, g],
    [J_x]
)

fn_cartpole_jacobian_u = ca.Function(
    'fn_cartpole_jacobian_u',
    [x, u, w, m_c, m_p, l, g],
    [J_u]
)

fn_cartpole_sim_step = ca.Function(
    'fn_cartpole_sim_step',
    [x, u, w, m_c, m_p, l, g, dt],
    [x_next]
)

# ============================================================
# 7. Quick sanity check
# ============================================================

x_test = ca.DM([0.0, 0.0, 0.1, 0.0])
u_test = ca.DM([0.0])
w_test = ca.DM([0.0, 0.0])

m_c_test = 1.0
m_p_test = 0.1
l_test   = 0.5
g_test   = 9.81
dt_test  = 0.01

x_next_test = fn_cartpole_sim_step(
    x_test, u_test, w_test,
    m_c_test, m_p_test, l_test, g_test, dt_test
)

print("x_next_test =", x_next_test)

# ============================================================
# 8. Save CasADi functions
# ============================================================

out_dir = os.path.join('src', 'casadi_functions')
os.makedirs(out_dir, exist_ok=True)

fn_cartpole_dynamics.save(os.path.join(out_dir, 'fn_cartpole_dynamics.casadi'))
fn_cartpole_jacobian_x.save(os.path.join(out_dir, 'fn_cartpole_jacobian_x.casadi'))
fn_cartpole_jacobian_u.save(os.path.join(out_dir, 'fn_cartpole_jacobian_u.casadi'))
fn_cartpole_sim_step.save(os.path.join(out_dir, 'fn_cartpole_sim_step.casadi'))

print("Saved cartpole CasADi functions to", out_dir)
