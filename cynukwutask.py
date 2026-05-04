"""
DEPLOYMENT-PROOF STREAMLIT PACKAGE
Thermal-Aware Spot-Based Scan Strategy (E-PBF)

Goal
----
This version is fully deployment-safe for Streamlit Cloud:
- NO matplotlib
- NO pandas
- ONLY streamlit + numpy
- Uses built-in Streamlit charts only

Conceptual link
---------------
This simulates scan-path planning in Electron Beam Powder Bed Fusion (E-PBF)
systems such as Freemelt ONE, where beam movement, thermal history,
and spot-based melting influence material quality.

It does NOT control hardware.
"""

import streamlit as st
import numpy as np

# =========================================================
# 1. PARAMETERS
# =========================================================

cooling_radius = 15.0
cooldown_steps = 5
heat_increase = 1.0
decay = 0.85
temp_threshold = 2.5

alpha = 1.0
beta = 2.0

# =========================================================
# 2. THERMAL MODEL
# =========================================================

def init_temperature(n):
    return np.zeros(n), np.full(n, -999)


def update_temperature(idx, points, temp):
    for i in range(len(points)):
        dist = np.linalg.norm(points[i] - points[idx])

        if dist < cooling_radius:
            temp[i] += heat_increase * (1.0 - dist / cooling_radius)

        temp[i] *= decay


def is_valid(i, step, temp, last):
    if temp[i] > temp_threshold:
        return False
    if step - last[i] < cooldown_steps:
        return False
    return True

# =========================================================
# 3. COST FUNCTION
# =========================================================

def cost(curr, cand, points, temp):
    return (
        alpha * np.linalg.norm(points[curr] - points[cand])
        + beta * temp[cand]
    )

# =========================================================
# 4. OPTIMIZER
# =========================================================

def run_optimizer(points):
    n = len(points)
    temp, last = init_temperature(n)

    path = [0]
    last[0] = 0
    update_temperature(0, points, temp)

    for step in range(1, n):
        curr = path[-1]

        candidates = [
            i for i in range(n)
            if i not in path and is_valid(i, step, temp, last)
        ]

        if not candidates:
            candidates = [i for i in range(n) if i not in path]

        nxt = min(candidates, key=lambda i: cost(curr, i, points, temp))

        path.append(nxt)
        last[nxt] = step
        update_temperature(nxt, points, temp)

    return path, temp

# =========================================================
# 5. OBP OUTPUT
# =========================================================

def dwell(t):
    return 1.0 + 0.5 * float(t)


def obp(path, points, temp):
    out = []
    for i in path:
        x, y = points[i]
        out.append(f"MOVE {x:.2f} {y:.2f}")
        out.append(f"EXPOSE {dwell(temp[i]):.2f}")
    return out

# =========================================================
# 6. STREAMLIT APP
# =========================================================

st.title("🔥 Thermal-Aware Scan Strategy (Deployment Proof)")

st.sidebar.header("Settings")

num_points = st.sidebar.slider("Scan points", 10, 200, 50, key="n")
seed = st.sidebar.number_input("Seed", value=42, key="s")

np.random.seed(int(seed))
points = np.random.rand(num_points, 2) * 100

path, temp = run_optimizer(points)
cmds = obp(path, points, temp)

# =========================================================
# 7. METRICS
# =========================================================

mean_temp = float(np.mean(temp))
max_temp = float(np.max(temp))
hot_points = int(np.sum(temp > temp_threshold))

st.subheader("📊 Results")

st.write({
    "Mean temperature": mean_temp,
    "Max temperature": max_temp,
    "Hot points": hot_points,
    "Path length": len(path)
})

# =========================================================
# 8. VISUALIZATION (NO MATPLOTLIB)
# =========================================================

st.subheader("🧭 Scan Path")
st.scatter_chart(points)

st.subheader("📈 Path Order (X over index)")
st.line_chart(points[path])

st.subheader("🌡️ Temperature Field")
st.line_chart(temp)

# =========================================================
# 9. OBP OUTPUT
# =========================================================

st.subheader("🤖 OBP Commands")
st.code("\n".join(cmds))

st.download_button(
    "Download OBP file",
    data="\n".join(cmds),
    file_name="obp_commands.txt",
    mime="text/plain",
    key="dl"
)

# =========================================================
# 10. INTERVIEW CONTEXT
# =========================================================

st.info(
    """
This model demonstrates a thermal-aware scan strategy inspired by E-PBF systems.
It reflects how scan path decisions influence heat accumulation in powder bed fusion.
Conceptually aligned with open-architecture systems like Freemelt ONE.
"""
)
"""
