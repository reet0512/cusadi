import casadi as ca
import os

# === 1. Define symbolic variables ===
# State: [p_x, p_z, theta, v_x, v_z, omega]
x_quad = ca.SX.sym('x_quad', 6, 1)

# Control: [T1, T2]
u_quad = ca.SX.sym('u_quad', 2, 1)

# Disturbance: [w_x, w_z]  (wind forces)
w_quad = ca.SX.sym('w_quad', 2, 1)

# Physical parameters
m = ca.SX.sym('m')   # mass
I = ca.SX.sym('I')   # moment of inertia
l = ca.SX.sym('l')   # half-arm length
g = ca.SX.sym('g')   # gravity

# Timestep
dt = ca.SX.sym('dt') # timestep

# === 2. Unpack state, control, disturbance ===
p_x, p_z, theta, v_x, v_z, omega = x_quad[0], x_quad[1], x_quad[2], x_quad[3], x_quad[4], x_quad[5]
T1, T2 = u_quad[0], u_quad[1]
w_x, w_z = w_quad[0], w_quad[1]

# Total thrust and torque
T_total = T1 + T2
tau = l * (T1 - T2)

# === 3. Continuous-time dynamics ===
# m * ddot{p}_x = -(T1 + T2) * sin(theta) + w_x
# m * ddot{p}_z =  (T1 + T2) * cos(theta) - m g + w_z
# I * ddot{theta} = l (T1 - T2)

a_x = (-(T_total) * ca.sin(theta) + w_x) / m
a_z = ((T_total) * ca.cos(theta) - m * g + w_z) / m
alpha = tau / I

f_quad = ca.vertcat(
    v_x,     # p_x_dot
    v_z,     # p_z_dot
    omega,   # theta_dot
    a_x,     # v_x_dot
    a_z,     # v_z_dot
    alpha    # omega_dot
)

# === 4. Jacobian w.r.t. state ===
J_x_quad = ca.jacobian(f_quad, x_quad)

# (Optional) Jacobian w.r.t. control if you want it:
J_u_quad = ca.jacobian(f_quad, u_quad)

# === 5. Semi-implicit Euler integration step ===
# Update velocities with accelerations, then positions with updated velocities
v_x_next = v_x + a_x * dt
v_z_next = v_z + a_z * dt
omega_next = omega + alpha * dt

p_x_next = p_x + v_x_next * dt
p_z_next = p_z + v_z_next * dt
theta_next = theta + omega_next * dt

x_next_quad = ca.vertcat(
    p_x_next,
    p_z_next,
    theta_next,
    v_x_next,
    v_z_next,
    omega_next
)

# === 6. Create CasADi Functions ===
# Continuous dynamics f(x, u, w, params)
fn_quad_dynamics = ca.Function(
    'fn_quad_dynamics',
    [x_quad, u_quad, w_quad, m, I, l, g],
    [f_quad]
)

# Jacobian w.r.t. state
fn_quad_jacobian_x = ca.Function(
    'fn_quad_jacobian_x',
    [x_quad, u_quad, w_quad, m, I, l, g],
    [J_x_quad]
)

# Jacobian w.r.t. control (optional but often useful)
fn_quad_jacobian_u = ca.Function(
    'fn_quad_jacobian_u',
    [x_quad, u_quad, w_quad, m, I, l, g],
    [J_u_quad]
)

# One simulation step x_{k+1} = f_d(x_k, u_k, w_k)
fn_quad_sim_step = ca.Function(
    'fn_quad_sim_step',
    [x_quad, u_quad, w_quad, m, I, l, g, dt],
    [x_next_quad]
)

# === sanity test at Python / CasADi level ===
x_test = ca.DM([0.0, 1.0, 0.2, 0.5, 0.0, 0.0])
u_test = ca.DM([4.905, 4.905])     # hover
w_test = ca.DM([0.5, 0.0])
m_test = 1.0
I_test = 0.05
l_test = 0.2
g_test = 9.81
dt_test = 0.01

x_next_test = fn_quad_sim_step(x_test, u_test, w_test, m_test, I_test, l_test, g_test, dt_test)
print("Python/CasADi x_next_test =", x_next_test)


# === 7. Save them ===
out_dir = os.path.join('src', 'casadi_functions')
os.makedirs(out_dir, exist_ok=True)

fn_quad_dynamics.save(os.path.join(out_dir, 'fn_quad_dynamics.casadi'))
fn_quad_jacobian_x.save(os.path.join(out_dir, 'fn_quad_jacobian_x.casadi'))
fn_quad_jacobian_u.save(os.path.join(out_dir, 'fn_quad_jacobian_u.casadi'))
fn_quad_sim_step.save(os.path.join(out_dir, 'fn_quad_sim_step.casadi'))

print("✅ Quadrotor CasADi functions saved to", out_dir)
