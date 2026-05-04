import streamlit as st
import numpy as np

# =========================================================
# 1. PHYSICAL / PROCESS PARAMETERS
# =========================================================

cooling_radius = 15.0
cooldown_steps = 5
heat_increase = 1.0
decay = 0.85
thermal_threshold = 2.5

alpha = 1.0   # spatial cost weight
beta = 2.0    # thermal cost weight

# =========================================================
# 2. INITIAL STATE
# =========================================================

# temperature field represents local thermal history
# last_visit tracks temporal spacing between melt spots

def init_state(n):
    temperature = np.zeros(n)
    last_visit = np.full(n, -999)
    return temperature, last_visit

# =========================================================
# 3. THERMAL MODEL (POINT HEAT SOURCE APPROXIMATION)
# =========================================================
# Paper-aligned interpretation:
# Each point acts as a discrete heat input (spot melting).
# Nearby points accumulate residual heat -> affects microstructure.


def update_temperature(idx, points, temp):
    for i in range(len(points)):
        dist = np.linalg.norm(points[i] - points[idx])

        # localized heat diffusion approximation
        if dist < cooling_radius:
            temp[i] += heat_increase * (1.0 - dist / cooling_radius)

        # thermal relaxation (cooling between scans)
        temp[i] *= decay

# =========================================================
# 4. CONSTRAINT MODEL (PROCESS STABILITY)
# =========================================================
# Prevents excessive local reheating (important for texture control)


def valid(i, step, temp, last_visit):
    if temp[i] > thermal_threshold:
        return False
    if step - last_visit[i] < cooldown_steps:
        return False
    return True

# =========================================================
# 5. COST FUNCTION (SCAN STRATEGY OPTIMIZATION)
# =========================================================
# Paper-aligned interpretation:
# Competing objectives:
# - minimize travel distance (efficiency)
# - minimize thermal accumulation (texture control)


def cost(curr, cand, points, temp):
    spatial = np.linalg.norm(points[curr] - points[cand])
    thermal = temp[cand]
    return alpha * spatial + beta * thermal

# =========================================================
# 6. TEXTURE METRIC (PAPER-INSPIRED EXTENSION)
# =========================================================
# Approximation of texture uniformity:
# Measures angular variability of scan directions


def texture_metric(points, path):
    if len(path) < 3:
        return 0.0

    angles = []

    for i in range(2, len(path)):
        p1 = points[path[i-2]]
        p2 = points[path[i-1]]
        p3 = points[path[i]]

        v1 = p2 - p1
        v2 = p3 - p2

        angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
        angles.append(angle)

    return float(np.std(angles))

# =========================================================
# 7. OPTIMIZER (SPOT MELTING STRATEGY)
# =========================================================


def run(points):
    n = len(points)
    temp, last_visit = init_state(n)

    path = [0]
    last_visit[0] = 0
    update_temperature(0, points, temp)

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
        update_temperature(nxt, points, temp)

    return path, temp

# =========================================================
# 8. OBP-LIKE MACHINE OUTPUT
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
# 9. STREAMLIT APP
# =========================================================

st.title("🔥 Paper-Aligned Spot Melting Scan Strategy")

st.sidebar.header("Parameters")

n = st.sidebar.slider("Number of points", 10, 200, 60, key="n")
seed = st.sidebar.number_input("Seed", value=42, key="seed")

np.random.seed(int(seed))
points = np.random.rand(n, 2) * 100

path, temp = run(points)
cmds = obp(path, points, temp)
tex = texture_metric(points, path)

# =========================================================
# 10. RESULTS
# =========================================================

st.subheader("Process Metrics")

st.write({
    "Mean temperature": float(np.mean(temp)),
    "Max temperature": float(np.max(temp)),
    "Hot spots": int(np.sum(temp > thermal_threshold)),
    "Texture variability (lower = more uniform)": tex,
    "Path length": len(path)
})

# =========================================================
# 11. VISUALIZATION
# =========================================================

st.subheader("Scan Path")
st.scatter_chart(points)

st.subheader("Thermal Field")
st.line_chart(temp)

st.subheader("Scan Sequence Projection")
st.line_chart(points[path])

# =========================================================
# 12. MACHINE OUTPUT
# =========================================================

st.subheader("OBP Commands")
st.code("\n".join(cmds))

st.download_button(
    "Download OBP file",
    data="\n".join(cmds),
    file_name="obp_commands.txt",
    mime="text/plain",
    key="dl"
)

# =========================================================
# 13. INDUSTRIAL INTERPRETATION
# =========================================================

st.info(
    "This implementation follows a spot-based melting strategy inspired by "
    "texture control approaches in metal additive manufacturing literature. "
    "It approximates how scan sequencing influences thermal gradients and microstructure evolution. "
    "Conceptually aligned with open-architecture E-PBF systems such as Freemelt ONE."
)
