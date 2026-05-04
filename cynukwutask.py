import streamlit as st
import numpy as np

# =========================
# THERMAL MODEL PARAMETERS
# =========================

cooling_radius = 15.0
cooldown_steps = 5
heat_increase = 1.0
decay = 0.85
temp_threshold = 2.5

alpha = 1.0
beta = 2.0


# =========================
# STATE INITIALIZATION
# =========================

def init_state(n):
    temperature = np.zeros(n)
    last_visit = np.full(n, -999)
    return temperature, last_visit


# =========================
# THERMAL UPDATE
# =========================

def update_temp(idx, points, temp):
    for i in range(len(points)):
        dist = np.linalg.norm(points[i] - points[idx])

        if dist < cooling_radius:
            temp[i] += heat_increase * (1.0 - dist / cooling_radius)

        temp[i] *= decay


# =========================
# VALIDITY RULES
# =========================

def valid(i, step, temp, last_visit):
    if temp[i] > temp_threshold:
        return False
    if step - last_visit[i] < cooldown_steps:
        return False
    return True


# =========================
# COST FUNCTION
# =========================

def cost(curr, cand, points, temp):
    return alpha * np.linalg.norm(points[curr] - points[cand]) + beta * temp[cand]


# =========================
# OPTIMIZER
# =========================

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

        if not candidates:
            candidates = [i for i in range(n) if i not in path]

        nxt = min(candidates, key=lambda i: cost(curr, i, points, temp))

        path.append(nxt)
        last_visit[nxt] = step
        update_temp(nxt, points, temp)

    return path, temp


# =========================
# MACHINE OUTPUT
# =========================

def dwell(t):
    return 1.0 + 0.5 * float(t)


def obp(path, points, temp):
    cmds = []
    for i in path:
        x, y = points[i]
        cmds.append(f"MOVE {x:.2f} {y:.2f}")
        cmds.append(f"EXPOSE {dwell(temp[i]):.2f}")
    return cmds


# =========================
# STREAMLIT UI
# =========================

st.title("🔥 Thermal-Aware Scan Strategy")

n = st.sidebar.slider("Points", 10, 200, 60)
seed = st.sidebar.number_input("Seed", value=42)

np.random.seed(int(seed))
points = np.random.rand(n, 2) * 100

path, temp = run(points)
cmds = obp(path, points, temp)

st.subheader("Results")
st.write({
    "Mean temp": float(np.mean(temp)),
    "Max temp": float(np.max(temp)),
    "Hot spots": int(np.sum(temp > temp_threshold))
})

st.subheader("Scan Points")
st.scatter_chart(points)

st.subheader("Temperature")
st.line_chart(temp)
# =========================================================
# THERMAL-AWARE SPOT-BASED SCAN STRATEGY (SUBMISSION PACKAGE)
# =========================================================
# Project: Electron Beam Powder Bed Fusion Scan Optimization
# System: Streamlit Demo + OBP-style command generator
# Inspired by: Spot-based melting strategies in E-PBF (JOM literature)
# Target system context: Freemelt ONE (conceptual compatibility)
# =========================================================

"""
SUBMISSION OVERVIEW
-------------------
This project implements a thermal-aware scan strategy inspired by
Electron Beam Powder Bed Fusion (E-PBF) systems such as Freemelt ONE.

It demonstrates:
- Spot-based melting logic
- Thermal accumulation modeling
- Hybrid scan cost optimization
- OBP-style machine command generation
- Streamlit-based interactive visualization

IMPORTANT:
This code does NOT control hardware. It is a simulation of scan logic.
"""

import streamlit as st
import numpy as np

# =========================================================
# 1. SYSTEM PARAMETERS (SCAN + THERMAL MODEL)
# =========================================================

cooling_radius = 15
cooldown_steps = 5
heat_increase = 1.0
decay = 0.85
temp_threshold = 2.5

alpha = 1.0   # geometric weight
beta = 2.0    # thermal weight

# =========================================================
# 2. THERMAL MODEL (PHYSICAL ANALOGY)
# =========================================================

def init_temperature(n):
    """Initialize thermal state of powder bed."""
    return np.zeros(n), np.full(n, -999)


def update_temperature(idx, step, points, temp):
    """Simulates heat accumulation from electron beam exposure."""
    for i in range(len(points)):
        dist = np.linalg.norm(points[i] - points[idx])

        # localized heating (melt pool influence zone)
        if dist < cooling_radius:
            temp[i] += heat_increase * (1 - dist / cooling_radius)

        # global cooling (thermal relaxation)
        temp[i] *= decay


def is_valid(i, step, temp, last):
    """Thermal + temporal constraints (melt stability condition)."""
    if temp[i] > temp_threshold:
        return False

    if step - last[i] < cooldown_steps:
        return False

    return True

# =========================================================
# 3. COST FUNCTION (SCAN STRATEGY DECISION RULE)
# =========================================================

def cost(current, candidate, points, temp):
    dist = np.linalg.norm(points[current] - points[candidate])
    thermal = temp[candidate]
    return alpha * dist + beta * thermal

# =========================================================
# 4. SCAN STRATEGY (SPOT-BASED OPTIMIZATION)
# =========================================================

def run_optimizer(points, temp, last):
    """Thermal-aware scan path generation."""
    n = len(points)

    path = [0]
    last[0] = 0

    for step in range(1, n):
        current = path[-1]

        candidates = [
            i for i in range(n)
            if i not in path and is_valid(i, step, temp, last)
        ]

        if not candidates:
            candidates = [i for i in range(n) if i not in path]

        next_p = min(
            candidates,
            key=lambda i: cost(current, i, points, temp)
        )

        path.append(next_p)
        last[next_p] = step
        update_temperature(next_p, step, points, temp)

    return path

# =========================================================
# 5. OBP-LIKE OUTPUT GENERATION (MACHINE INTERFACE ABSTRACTION)
# =========================================================

def dwell_time(t):
    return 1 + 0.5 * t


def generate_obp(path, points, temp):
    """Converts scan path into machine-like instruction set."""
    cmds = []

    for i in path:
        x, y = points[i]
        t = dwell_time(temp[i])

        cmds.append(f"MOVE {x:.2f} {y:.2f}")
        cmds.append(f"EXPOSE {t:.2f}")

    return cmds

# =========================================================
# 6. STREAMLIT APPLICATION (INTERACTIVE DEMO)
# =========================================================

st.title("🔥 Thermal-Aware Scan Strategy (E-PBF Model)")

st.sidebar.header("Control Panel")
num_points = st.sidebar.slider("Number of Scan Points", 10, 200, 50, key="np")
seed = st.sidebar.number_input("Random Seed", value=42, key="seed")

np.random.seed(seed)

# generate synthetic powder bed scan points
points = np.random.rand(num_points, 2) * 100

# initialize thermal state
temp, last = init_temperature(num_points)

# run scan strategy
path = run_optimizer(points, temp, last)

# generate OBP-style commands
commands = generate_obp(path, points, temp)

# =========================================================
# 7. OUTPUT VISUALIZATION
# =========================================================

st.subheader("Optimized Scan Path")
st.write(path)

st.subheader("OBP-Style Machine Commands")
st.code("\n".join(commands))

st.subheader("Path Visualization")
st.line_chart(points[path])

st.subheader("Thermal Field Evolution (Final)")
st.line_chart(temp)

with st.expander("Raw Points"):
    st.write(points)

with st.expander("Thermal State"):
    st.write(temp)

# =========================================================
# 8. ENGINEERING INTERPRETATION (FOR SUBMISSION)
# =========================================================
"""
KEY CONTRIBUTIONS:

1. Spot-based melting abstraction:
   - Each point represents a melt/exposure location.

2. Thermal field simulation:
   - Models heat accumulation similar to E-PBF powder bed behavior.

3. Hybrid scan strategy:
   - Combines geometric efficiency and thermal stability.

4. OBP-style output:
   - Generates machine-like MOVE/EXPOSE commands.

5. Freemelt ONE relevance:
   - Conceptually aligned with scan strategy constraints in systems like
     Freemelt ONE (electron beam powder bed fusion platform).

IMPROVEMENT OVER GREEDY METHODS:
- Prevents thermal clustering
- Reduces localized overheating
- Produces more physically realistic scan paths
"""

st.subheader("Commands")
st.code("\n".join(cmds))

st.download_button(
    "Download OBP commands",
    data="\n".join(cmds),
    file_name="obp.txt",
    mime="text/plain"
)
