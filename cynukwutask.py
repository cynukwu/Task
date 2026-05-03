import streamlit as st
import numpy as np

# =========================
# CONFIG PARAMETERS
# =========================

cooling_radius = 15
cooldown_steps = 5
heat_increase = 1.0
decay = 0.85
temp_threshold = 2.5

alpha = 1.0   # distance weight
beta = 2.0    # thermal weight

# =========================
# THERMAL MODEL
# =========================

def init_temperature(num_points):
    temperature = np.zeros(num_points)
    last_visited_step = np.full(num_points, -999)
    return temperature, last_visited_step


def update_temperature(visited_idx, step, points, temperature):
    for i in range(len(points)):
        dist = np.linalg.norm(points[i] - points[visited_idx])

        # spatial heating
        if dist < cooling_radius:
            temperature[i] += heat_increase * (1 - dist / cooling_radius)

        # global decay
        temperature[i] *= decay


def is_valid(candidate, step, temperature, last_visited_step):
    if temperature[candidate] > temp_threshold:
        return False

    if step - last_visited_step[candidate] < cooldown_steps:
        return False

    return True

# =========================
# COST FUNCTION
# =========================

def cost(current, candidate, points, temperature):
    dist_cost = np.linalg.norm(points[current] - points[candidate])
    thermal_cost = temperature[candidate]
    return alpha * dist_cost + beta * thermal_cost

# =========================
# OPTIMIZER
# =========================

def run_optimizer(points, temperature, last_visited_step):
    num_points = len(points)

    visited = [0]
    last_visited_step[0] = 0

    for step in range(1, num_points):
        current = visited[-1]

        candidates = [
            i for i in range(num_points)
            if i not in visited and is_valid(i, step, temperature, last_visited_step)
        ]

        if not candidates:
            candidates = [i for i in range(num_points) if i not in visited]

        next_point = min(
            candidates,
            key=lambda i: cost(current, i, points, temperature)
        )

        visited.append(next_point)
        last_visited_step[next_point] = step
        update_temperature(next_point, step, points, temperature)

    return visited

# =========================
# STREAMLIT APP
# =========================

st.title("🔥 Thermal Path Optimizer")

st.sidebar.header("Settings")
num_points = st.sidebar.slider("Number of Points", 10, 200, 50)
seed = st.sidebar.number_input("Random Seed", value=42)

np.random.seed(seed)

# Generate points
points = np.random.rand(num_points, 2) * 100

# Init system
temperature, last_visited_step = init_temperature(num_points)

# Run optimizer
path = run_optimizer(points, temperature, last_visited_step)

# =========================
# OUTPUT
# =========================

st.subheader("Optimized Path")
st.write(path)

# Plot path
path_points = points[path]

st.subheader("Path Visualization")
chart_data = np.array(path_points)
st.line_chart(chart_data)

# Temperature visualization
st.subheader("Temperature Field (final)")
st.line_chart(temperature)

# Show raw data
with st.expander("Raw Points"):
    st.write(points)

with st.expander("Temperature Array"):
    st.write(temperature)
