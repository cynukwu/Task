# =========================================================
# CLEAN PRODUCTION VERSION (STREAMLIT SAFE)
# Thermal-Aware Scan Strategy for E-PBF
# =========================================================

import streamlit as st
import numpy as np

# =========================================================
# PARAMETERS
# =========================================================

cooling_radius = 15.0
cooldown_steps = 5
heat_increase = 1.0
decay = 0.85
temp_threshold = 2.5

alpha = 1.0
beta = 2.0

# =========================================================
# THERMAL MODEL
# =========================================================

def init_state(n):
    temperature = np.zeros(n)
    last_visit = np.full(n, -999)
    return temperature, last_visit


def update_temp(idx, points, temp):
    for i in range(len(points)):
        dist = np.linalg.norm(points[i] - points[idx])

        if dist < cooling_radius:
            temp[i] += heat_increase * (1.0 - dist / cooling_radius)

        temp[i] *= decay


def valid(i, step, temp, last_visit):
    if temp[i] > temp_threshold:
        return False
    if step - last_visit[i] < cooldown_steps:
        return False
    return True

# =========================================================
# COST FUNCTION
# =========================================================

def cost(curr, cand, points, temp):
    dist = np.linalg.norm(points[curr] - points[cand])
    thermal = temp[cand]
    return alpha * dist + beta * thermal

# =========================================================
# OPTIMIZER
# =========================================================

def run(points):
    n = len(points)
    temp, last_visit = init_state(n)

    path = [0]
    last_visit[0] = 0
    update_temp(0, points, temp)

    for step in range(1, n):
        curr = path[-1]

        candidates = [i for i in range(n) if i not in path and valid(i, step, temp, last_visit)]

        if not candidates:
            candidates = [i for i in range(n) if i not in path]

        nxt = min(candidates, key=lambda i: cost(curr, i, points, temp))

        path.append(nxt)
        last_visit[nxt] = step
        update_temp(nxt, points, temp)

    return path, temp

# =========================================================
# OBP OUTPUT
# =========================================================

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
# STREAMLIT APP
# =========================================================

st.title("Thermal-Aware Scan Strategy (Clean Production Version)")

st.sidebar.header("Settings")

n = st.sidebar.slider("Number of points", 10, 200, 50, key="n")
seed = st.sidebar.number_input("Seed", value=42, key="seed")

np.random.seed(int(seed))
points = np.random.rand(n, 2) * 100

path, temp = run(points)
cmds = obp(path, points, temp)

# =========================================================
# METRICS
# =========================================================

st.subheader("Results")

st.write({
    "Path length": len(path),
    "Mean temperature": float(np.mean(temp)),
    "Max temperature": float(np.max(temp)),
    "Hot points": int(np.sum(temp > temp_threshold))
})

# =========================================================
# VISUALIZATION
# =========================================================

st.subheader("Scan Points")
st.scatter_chart(points)

st.subheader("Temperature Field")
st.line_chart(temp)

st.subheader("Path Evolution")
st.line_chart(points[path])

# =========================================================
# OBP OUTPUT
# =========================================================

st.subheader("Machine Commands (OBP-style)")
st.code("\n".join(cmds))

st.download_button(
    "Download commands",
    data="\n".join(cmds),
    file_name="obp.txt",
    mime="text/plain",
    key="download"
)

# =========================================================
# INTERVIEW NOTE
# =========================================================

st.info(
    "Thermal-aware scan strategy simulating E-PBF beam path planning. "
    "Uses hybrid cost function combining geometry and thermal hist
