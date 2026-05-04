import streamlit as st
import numpy as np

"""
Thermal Path Optimizer for Streamlit

What this app does:
- Generates a set of 2D scan points.
- Chooses the next point using a hybrid cost function.
- Tracks a simple thermal field so nearby points become less desirable.
- Displays the resulting path and the final temperature values.

Connection to Freemelt ONE:
This code is a simplified, educational model inspired by thermal-aware planning in
Electron Beam Powder Bed Fusion (E-PBF). Freemelt ONE is an E-PBF system with a
6 kW electron gun, high powder bed temperatures, and a protected chamber designed
for hot processing. Freemelt also describes ProHeat, a preheating method used across
its machine models, and line melting / hatching as a common E-PBF strategy.
This script does not control the machine; it only demonstrates the logic behind
spacing, cooling, and thermal-aware path selection.
"""

# =========================
# CONFIG PARAMETERS
# =========================
# These values define the thermal behavior of the model.
# They are tunable knobs for the assignment and for experiments.

cooling_radius = 15     # Nearby points within this distance receive heat influence.
cooldown_steps = 5      # A point must wait this many steps before being valid again.
heat_increase = 1.0     # How much heat is added to nearby points.
decay = 0.85            # Global cooling after every update.
temp_threshold = 2.5    # Above this temperature, a point is considered too hot.

alpha = 1.0             # Weight for geometric distance in the cost function.
beta = 2.0              # Weight for thermal penalty in the cost function.

# =========================
# THERMAL MODEL
# =========================
# This block simulates a simple heat field.
# In a real E-PBF setting, the heat field would be more complex, but the idea is
# the same: recent exposure influences nearby areas and changes future decisions.


def init_temperature(num_points):
    """Create the initial thermal state for all points."""
    temperature = np.zeros(num_points)
    last_visited_step = np.full(num_points, -999)
    return temperature, last_visited_step


def update_temperature(visited_idx, step, points, temperature):
    """
    Update the temperature field after visiting one point.

    Function:
    - Adds heat to nearby points.
    - Applies global decay.

    Use:
    - Prevents repeated scanning of the same hot region.

    Improvement applied:
    - Replaces a pure distance-only greedy method with a heat-aware model.
    """
    for i in range(len(points)):
        dist = np.linalg.norm(points[i] - points[visited_idx])

        # Spatial heating: points close to the visited point get extra heat.
        if dist < cooling_radius:
            temperature[i] += heat_increase * (1 - dist / cooling_radius)

        # Global decay: the whole field cools down over time.
        temperature[i] *= decay


def is_valid(candidate, step, temperature, last_visited_step):
    """
    Decide whether a candidate point is allowed.

    Function:
    - Rejects points that are too hot.
    - Rejects points that were visited too recently.

    Use:
    - Enforces spacing in both temperature and time.

    Improvement applied:
    - Adds a cooling constraint so the path does not bounce around one hot zone.
    """
    if temperature[candidate] > temp_threshold:
        return False

    if step - last_visited_step[candidate] < cooldown_steps:
        return False

    return True

# =========================
# COST FUNCTION
# =========================
# This is the main decision rule.
# The next point is chosen by balancing travel distance and thermal penalty.


def cost(current, candidate, points, temperature):
    """
    Hybrid cost = distance cost + thermal cost.

    Function:
    - Measures how far the candidate is.
    - Penalizes candidates in hotter areas.

    Use:
    - Makes the optimizer prefer cooler and closer points.

    Improvement applied:
    - Better than nearest-neighbor alone because it avoids thermal clustering.
    """
    dist_cost = np.linalg.norm(points[current] - points[candidate])
    thermal_cost = temperature[candidate]
    return alpha * dist_cost + beta * thermal_cost

# =========================
# OPTIMIZER
# =========================
# This is the path builder.
# It starts at point 0 and keeps selecting the best next valid point.


def run_optimizer(points, temperature, last_visited_step):
    """
    Build a thermally aware path.

    Function:
    - Starts at the first point.
    - Filters valid candidates.
    - Chooses the candidate with the lowest hybrid cost.
    - Updates the thermal field after every move.

    Use:
    - Produces a scan route that spreads heat more evenly.

    Improvement applied:
    - Compared with a plain greedy route, this version is more physically realistic.
    """
    num_points = len(points)

    visited = [0]
    last_visited_step[0] = 0

    for step in range(1, num_points):
        current = visited[-1]

        # Only consider points not yet visited and that pass the thermal rules.
        candidates = [
            i for i in range(num_points)
            if i not in visited and is_valid(i, step, temperature, last_visited_step)
        ]

        # Fallback: if everything is blocked, relax the thermal constraint.
        if not candidates:
            candidates = [i for i in range(num_points) if i not in visited]

        # Choose the point with the lowest combined geometric + thermal cost.
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
# This is the user interface.
# It lets you run the optimizer and see the output without changing the code.

st.title("🔥 Thermal Path Optimizer")

st.sidebar.header("Settings")
num_points = st.sidebar.slider("Number of Points", 10, 200, 50)
seed = st.sidebar.number_input("Random Seed", value=42)

# Fixed seed makes the example reproducible.
np.random.seed(seed)

# Generate a synthetic 2D point set in a 100x100 workspace.
points = np.random.rand(num_points, 2) * 100

# Initialize the thermal state.
temperature, last_visited_step = init_temperature(num_points)

# Run the optimizer.
path = run_optimizer(points, temperature, last_visited_step)

# =========================
# OUTPUT
# =========================

st.subheader("Optimized Path")
st.write(path)

# Path visualization.
# A true scatter/line plot would be better for publication,
# but this keeps the example simple and Streamlit-friendly.
path_points = points[path]

st.subheader("Path Visualization")
chart_data = np.array(path_points)
st.line_chart(chart_data)

# Final temperature field.
# This shows where the model built up heat by the end of the scan.
st.subheader("Temperature Field (final)")
st.line_chart(temperature)

# Raw data is hidden inside expanders to keep the page clean.
with st.expander("Raw Points"):
    st.write(points)

with st.expander("Temperature Array"):
    st.write(temperature)
