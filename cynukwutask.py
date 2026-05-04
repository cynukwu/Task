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

st.subheader("Commands")
st.code("\n".join(cmds))

st.download_button(
    "Download OBP commands",
    data="\n".join(cmds),
    file_name="obp.txt",
    mime="text/plain"
)
