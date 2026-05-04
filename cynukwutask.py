# =========================================================
# THERMAL-AWARE SCAN STRATEGY (STREAMLIT SAFE)
# WITH INLINE EXPLANATIONS FOR INTERVIEW UNDERSTANDING
# =========================================================

import streamlit as st
import numpy as np

# =========================================================
# WHAT THIS PROJECT DOES (HIGH LEVEL)
# =========================================================
# This app simulates a simplified scan strategy used in
# electron beam powder bed fusion (E-PBF).
#
# It models:
# - Spot-based energy deposition (each point = melt event)
# - Heat accumulation in nearby regions
# - Cooling over time
# - Scan path optimization using a hybrid rule
#
# This is NOT machine control software.
# It is an educational physics-inspired optimization model.
# =========================================================

# =========================================================
# 1. PARAMETERS (CONTROLLING THE PHYSICS)
# =========================================================
# These parameters define how heat spreads and decays.
# They also control how strictly we avoid overheating regions.

cooling_radius = 15.0      # how far heat spreads spatially
cooldown_steps = 5         # how long before revisiting same region
heat_increase = 1.0        # strength of each spot heat input
decay = 0.85               # global cooling factor
temp_threshold = 2.5       # max allowed temperature

# Cost function weights:
# alpha = importance of travel distance
# beta = importance of avoiding hot zones

alpha = 1.0
beta = 2.0

# =========================================================
# 2. STATE INITIALIZATION
# =========================================================
# We store:
# - temperature field (thermal history)
# - last visit time (temporal spacing constraint)


def init_state(n):
    temperature = np.zeros(n)
    last_visit = np.full(n, -999)
    return temperature, last_visit

# =========================================================
# 3. THERMAL UPDATE MODEL
# =========================================================
# Each time we visit a point:
# - nearby points receive heat
# - all points cool down slightly
#
# This mimics localized energy input in E-PBF spot melting.


def update_temp(idx, points, temp):
    for i in range(len(points)):
        dist = np.linalg.norm(points[i] - points[idx])

        # local heat influence (Gaussian-like simplification)
        if dist < cooling_radius:
            temp[i] += heat_increase * (1.0 - dist / cooling_radius)

        # global cooling (thermal relaxation)
        temp[i] *= decay

# =========================================================
# 4. VALIDITY CHECK (PROCESS CONSTRAINT)
# =========================================================
# Prevents selecting points that are:
# - too hot (thermal overload)
# - too recently visited (cooling constraint)


def valid(i, step, temp, last_visit):
    if temp[i] > temp_threshold:
        return False
    if step - last_visit[i] < cooldown_steps:
        return False
    return True

# =========================================================
# 5. COST FUNCTION (HOW WE CHOOSE NEXT POINT)
# =========================================================
# The algorithm balances two competing goals:
# 1. Short travel distance (efficiency)
# 2. Low thermal risk (stability)


def cost(curr, cand, points, temp):
    dist = np.linalg.norm(points[curr] - points[cand])
    thermal = temp[cand]
    return alpha * dist + beta * thermal

# =========================================================
# 6. OPTIMIZER (MAIN PATH GENERATION)
# =========================================================
# This builds the scan path step by step.
# At each step:
# - filter valid candidates
# - choose best according to cost function
# - update thermal field


def run(points):
    n = len(points)
    temp, last_visit = init_state(n)

    path = [0]
    last_visit[0] = 0
    update_temp(0, points, temp)

    for step in range(1, n):
        curr = path[-1]

        candidates = [
            i for i in range(n)
            if i not in path and valid(i, step, temp, last_visit)
        ]

        # fallback: if all are blocked, allow remaining points
        if not candidates:
            candidates = [i for i in range(n) if i not in path]

        nxt = min(candidates, key=lambda i: cost(curr, i, points, temp))

        path.append(nxt)
        last_visit[nxt] = step
        update_temp(nxt, points, temp)

    return path, temp

# =========================================================
# 7. MACHINE OUTPUT (OBP-STYLE COMMANDS)
# =========================================================
# Converts scan path into simple machine-like instructions:
# MOVE = beam position
# EXPOSE = energy deposition time (depends on temperature)


def dwell(t):
    return 1.0 + 0.5 * float(t)


def obp(path, points, temp):
    cmds = []
    for i in path:
        x, y = points[i]
        cmds.append(f"MOVE {x:.2f} {y:.2f}")
        cmds.append(f"EXPOSE {dwell(temp[i]):.2f}")
    return cmds

# =========================================================
# 8. STREAMLIT USER INTERFACE
# =========================================================

st.title("Thermal-Aware Scan Strategy Simulator")

st.sidebar.header("Controls")

n = st.sidebar.slider("Number of points", 10, 200, 60, key="n")
seed = st.sidebar.number_input("Random seed", value=42, key="seed")

np.random.seed(int(seed))
points = np.random.rand(n, 2) * 100

# run model
path, temp = run(points)
cmds = obp(path, points, temp)

# =========================================================
# 9. OUTPUT METRICS (WHAT YOU EXPLAIN IN INTERVIEW)
# =========================================================
# These values show system behavior:
# - how hot the system becomes
# - how many constrained regions exist

st.subheader("System Metrics")

st.write({
    "Mean temperature": float(np.mean(temp)),
    "Max temperature": float(np.max(temp)),
    "Hot points": int(np.sum(temp > temp_threshold)),
    "Total steps": len(path)
})

# =========================================================
# 10. VISUALIZATION (STREAMLIT NATIVES ONLY)
# =========================================================

st.subheader("Spatial Layout")
st.scatter_chart(points)

st.subheader("Thermal Evolution")
st.line_chart(temp)

st.subheader("Scan Sequence")
st.line_chart(points[path])

# =========================================================
# 11. MACHINE COMMAND OUTPUT
# =========================================================

st.subheader("Generated Machine Instructions")
st.code("\n".join(cmds))

st.download_button(
    "Download commands",
    data="\n".join(cmds),
    file_nam
