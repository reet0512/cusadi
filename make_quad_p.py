# quadrotor_dynamics.py
import casadi as ca
import os

# ----------------------------------------------------------
# Quadrotor nonlinear 2D dynamics (matching the PDF exactly)
# ----------------------------------------------------------

# States: [p_x, p_z, v_x, v_z, theta, omega]
x = ca.SX.sym("x", 6)

# Controls: [T1, T2]
u = ca.SX.sym("u", 2)

# Wind disturbances: [w_x, w_z]  (ONLY 2 — exactly as PDF)
w = ca.SX.sym("w", 2)

# Parameters
m  = ca.SX.sym("m")
I  = ca.SX.sym("I")
l  = ca.SX.sym("l")
g  = ca.SX.sym("g")
dt = ca.SX.sym("dt")

# ----------------------------------------------------------
# Dynamics from PDF
# ----------------------------------------------------------
p_x, p_z, v_x, v_z, theta, omega = x[0], x[1], x[2], x[3], x[4], x[5]
T1, T2 = u[0], u[1]
w_x, w_z = w[0], w[1]

T = T1 + T2               # total thrust
M = (T1 - T2) * l         # torque

p_x_dot = v_x
p_z_dot = v_z

v_x_dot = -(T/m) * ca.sin(theta) + w_x
v_z_dot = -(T/m) * ca.cos(theta) - g + w_z

theta_dot = omega
omega_dot = M/I

x_dot = ca.vertcat(
    p_x_dot,
    p_z_dot,
    v_x_dot,
    v_z_dot,
    theta_dot,
    omega_dot
)

# Euler integration
x_next = x + dt * x_dot

# Save function
fn = ca.Function("fn_quad_step", [x, u, w, m, I, l, g, dt], [x_next])

os.makedirs("src/casadi_functions", exist_ok=True)
fn.save("src/casadi_functions/fn_quad_step.casadi")

print("✓ Saved fn_quad_step.casadi (PDF-accurate quadrotor dynamics)")
