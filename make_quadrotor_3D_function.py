import casadi as ca
import os

# === 1. Define symbolic variables ===
# State: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
x_quad = ca.SX.sym('x_quad', 12, 1)

# Control: [T1, T2, T3, T4]
u_quad = ca.SX.sym('u_quad', 4, 1)

# Disturbance: [w_x, w_y, w_z] (wind forces in world frame)
w_quad = ca.SX.sym('w_quad', 3, 1)

# Physical parameters
m   = ca.SX.sym('m')      # mass
Jx  = ca.SX.sym('Jx')     # inertia about x
Jy  = ca.SX.sym('Jy')     # inertia about y
Jz  = ca.SX.sym('Jz')     # inertia about z
l   = ca.SX.sym('l')      # arm length
k_m = ca.SX.sym('k_m')    # yaw torque coefficient
g   = ca.SX.sym('g')      # gravity

# Timestep
dt = ca.SX.sym('dt')      # timestep

# === 2. Unpack state, control, disturbance ===
px, py, pz   = x_quad[0], x_quad[1], x_quad[2]
vx, vy, vz   = x_quad[3], x_quad[4], x_quad[5]
phi, th, psi = x_quad[6], x_quad[7], x_quad[8]
p, q, r      = x_quad[9], x_quad[10], x_quad[11]

T1, T2, T3, T4 = u_quad[0], u_quad[1], u_quad[2], u_quad[3]
w_x, w_y, w_z  = w_quad[0], w_quad[1], w_quad[2]

# Total thrust
T_total = T1 + T2 + T3 + T4

# Torques (standard quad layout, simplified)
tau_phi   = l * (T2 - T4)
tau_theta = l * (T3 - T1)
tau_psi   = k_m * (T1 - T2 + T3 - T4)

# === 3. Rotation matrix R_body_to_world (ZYX: yaw, pitch, roll) ===
cphi = ca.cos(phi); sphi = ca.sin(phi)
cth  = ca.cos(th);  sth  = ca.sin(th)
cpsi = ca.cos(psi); spsi = ca.sin(psi)

Rz = ca.vertcat(
    ca.hcat([cpsi, -spsi, 0]),
    ca.hcat([spsi,  cpsi, 0]),
    ca.hcat([0,     0,    1])
)

Ry = ca.vertcat(
    ca.hcat([cth, 0, sth]),
    ca.hcat([0,   1, 0  ]),
    ca.hcat([-sth,0, cth])
)

Rx = ca.vertcat(
    ca.hcat([1,    0,     0   ]),
    ca.hcat([0,  cphi, -sphi]),
    ca.hcat([0,  sphi,  cphi])
)

R = Rz @ Ry @ Rx

# Thrust in body frame is along +z_b
thrust_b = ca.vertcat(0, 0, T_total)
thrust_w = R @ thrust_b

# === 4. Continuous-time dynamics ===

# Translational acceleration
a_x = (thrust_w[0] + w_x) / m
a_y = (thrust_w[1] + w_y) / m
a_z = (thrust_w[2] + w_z) / m - g

# Attitude kinematics (Euler ZYX, body rates p,q,r)
#   [phi_dot]   [ 1,  sin(phi)*tan(theta),  cos(phi)*tan(theta)] [p]
#   [th_dot ] = [ 0,          cos(phi),            -sin(phi)    ] [q]
#   [psi_dot]   [ 0,  sin(phi)/cos(theta), cos(phi)/cos(theta) ] [r]

# Protect against cos(theta) ~ 0, but assume small angles here
phi_dot = p + q * sphi * ca.tan(th) + r * cphi * ca.tan(th)
th_dot  =     q * cphi             - r * sphi
psi_dot =     q * sphi / cth      + r * cphi / cth

# Angular acceleration (simplified diagonal inertia, no coupling)
p_dot = tau_phi   / Jx
q_dot = tau_theta / Jy
r_dot = tau_psi   / Jz

# === 5. Pack continuous-time dynamics f(x, u, w) ===
f_quad = ca.vertcat(
    vx,        # x_dot
    vy,        # y_dot
    vz,        # z_dot
    a_x,       # vx_dot
    a_y,       # vy_dot
    a_z,       # vz_dot
    phi_dot,   # phi_dot
    th_dot,    # theta_dot
    psi_dot,   # psi_dot
    p_dot,     # p_dot
    q_dot,     # q_dot
    r_dot      # r_dot
)

# === 6. Jacobians ===
J_x_quad = ca.jacobian(f_quad, x_quad)
J_u_quad = ca.jacobian(f_quad, u_quad)

# === 7. Semi-implicit Euler integration step ===
# Update velocities and angular rates, then positions and angles
vx_next = vx + a_x * dt
vy_next = vy + a_y * dt
vz_next = vz + a_z * dt

p_next = p + p_dot * dt
q_next = q + q_dot * dt
r_next = r + r_dot * dt

px_next = px + vx_next * dt
py_next = py + vy_next * dt
pz_next = pz + vz_next * dt

phi_next  = phi + phi_dot * dt
th_next   = th  + th_dot  * dt
psi_next  = psi + psi_dot * dt

x_next_quad = ca.vertcat(
    px_next,
    py_next,
    pz_next,
    vx_next,
    vy_next,
    vz_next,
    phi_next,
    th_next,
    psi_next,
    p_next,
    q_next,
    r_next
)

# === 8. Create CasADi Functions ===

fn_quad_dynamics_3d = ca.Function(
    'fn_quad_dynamics_3d',
    [x_quad, u_quad, w_quad, m, Jx, Jy, Jz, l, k_m, g],
    [f_quad]
)

fn_quad_jacobian_x_3d = ca.Function(
    'fn_quad_jacobian_x_3d',
    [x_quad, u_quad, w_quad, m, Jx, Jy, Jz, l, k_m, g],
    [J_x_quad]
)

fn_quad_jacobian_u_3d = ca.Function(
    'fn_quad_jacobian_u_3d',
    [x_quad, u_quad, w_quad, m, Jx, Jy, Jz, l, k_m, g],
    [J_u_quad]
)

fn_quad_sim_step_3d = ca.Function(
    'fn_quad_sim_step_3d',
    [x_quad, u_quad, w_quad, m, Jx, Jy, Jz, l, k_m, g, dt],
    [x_next_quad]
)

# === 9. Sanity test at Python / CasADi level ===
x_test = ca.DM([
    0.0, 0.0, 1.0,   # position x,y,z
    0.0, 0.0, 0.0,   # velocity
    0.0, 0.0, 0.0,   # phi, theta, psi
    0.0, 0.0, 0.0    # p, q, r
])

m_test   = 1.0
Jx_test  = 0.02
Jy_test  = 0.02
Jz_test  = 0.04
l_test   = 0.2
k_m_test = 0.01
g_test   = 9.81
dt_test  = 0.01

# hover thrust: split equally across 4 motors
T_hover = m_test * g_test
T_each  = T_hover / 4.0

u_test = ca.DM([T_each, T_each, T_each, T_each])
w_test = ca.DM([0.0, 0.0, 0.0])

x_next_test = fn_quad_sim_step_3d(
    x_test, u_test, w_test,
    m_test, Jx_test, Jy_test, Jz_test,
    l_test, k_m_test, g_test, dt_test
)
print("Python/CasADi 3D x_next_test =", x_next_test)

# === 10. Save functions ===
out_dir = os.path.join('src', 'casadi_functions')
os.makedirs(out_dir, exist_ok=True)

fn_quad_dynamics_3d.save(os.path.join(out_dir, 'fn_quad_dynamics_3d.casadi'))
fn_quad_jacobian_x_3d.save(os.path.join(out_dir, 'fn_quad_jacobian_x_3d.casadi'))
fn_quad_jacobian_u_3d.save(os.path.join(out_dir, 'fn_quad_jacobian_u_3d.casadi'))
fn_quad_sim_step_3d.save(os.path.join(out_dir, 'fn_quad_sim_step_3d.casadi'))

print("✅ 3D Quadrotor CasADi functions saved to", out_dir)
