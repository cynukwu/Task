"""
Thermal-Aware Spot-Based Scan Strategy for E-PBF
Streamlit submission package

Purpose
-------
This single-file app demonstrates a simplified, interview-ready scan strategy
for Electron Beam Powder Bed Fusion (E-PBF).

What it shows
-------------
1. Spot-based scan points in a 2D build area
2. Thermal accumulation and decay
3. A hybrid cost function that balances distance and thermal penalty
4. A baseline nearest-neighbor comparison
5. OBP-style MOVE / EXPOSE command generation
6. Streamlit visualization and downloadable output

Industrial framing
------------------
This is a simulation of scan-strategy logic, not machine control software.
It is conceptually aligned with open-architecture E-PBF workflows used in
systems such as Freemelt ONE, which is described by Freemelt as an E-PBF
platform with a 6 kW electron gun, high powder-bed temperatures, vacuum
operation, and a protected chamber for hot processing. Freemelt also describes
ProHeat preheating technology and spot melting / hatching strategies in its
materials and knowledge pages. The project also follows the idea of OBP-style
command generation discussed in the Freemelt Openmelt / OBPlib-Python GitLab
repository.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1) PARAMETERS
# =========================================================
# These are the main knobs used in the assignment.
# They let you explain how the model reacts to heat and how strongly it
# prioritizes geometric efficiency versus thermal stability.


@dataclass
class ThermalParams:
    cooling_radius: float = 15.0
    cooldown_steps: int = 5
    heat_increase: float = 1.0
    decay: float = 0.85
    temp_threshold: float = 2.5
    alpha: float = 1.0  # distance weight
    beta: float = 2.0   # thermal weight


# =========================================================
# 2) THERMAL MODEL
# =========================================================
# Function: create the thermal state, update heat, and filter hot candidates.
# Use: approximate heat accumulation after electron beam exposure.
# Improvement: introduces a thermal memory so the scan strategy is no longer
# purely distance based.


def init_temperature(num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    temperature = np.zeros(num_points, dtype=float)
    last_visited_step = np.full(num_points, -999, dtype=int)
    return temperature, last_visited_step


def update_temperature(
    visited_idx: int,
    points: np.ndarray,
    temperature: np.ndarray,
    params: ThermalParams,
) -> None:
    for i in range(len(points)):
        dist = np.linalg.norm(points[i] - points[visited_idx])

        # Nearby points receive heat from the exposure.
        if dist < params.cooling_radius:
            temperature[i] += params.heat_increase * (1.0 - dist / params.cooling_radius)

        # Global decay models cooling over time.
        temperature[i] *= params.decay


def is_valid(
    candidate: int,
    step: int,
    temperature: np.ndarray,
    last_visited_step: np.ndarray,
    params: ThermalParams,
) -> bool:
    if temperature[candidate] > params.temp_threshold:
        return False
    if step - last_visited_step[candidate] < params.cooldown_steps:
        return False
    return True


# =========================================================
# 3) COST FUNCTIONS
# =========================================================
# Function: score each candidate.
# Use: choose the next scan point.
# Improvement: combines travel distance and thermal penalty.


def hybrid_cost(current: int, candidate: int, points: np.ndarray, temperature: np.ndarray, params: ThermalParams) -> float:
    dist_cost = float(np.linalg.norm(points[current] - points[candidate]))
    thermal_cost = float(temperature[candidate])
    return params.alpha * dist_cost + params.beta * thermal_cost


def distance_only_cost(current: int, candidate: int, points: np.ndarray) -> float:
    return float(np.linalg.norm(points[current] - points[candidate]))


# =========================================================
# 4) OPTIMIZERS
# =========================================================
# Function: build a scan path.
# Use: compare a thermal-aware strategy against a baseline greedy strategy.
# Improvement: the thermal-aware version avoids repeatedly targeting hot areas.


def run_thermal_optimizer(points: np.ndarray, params: ThermalParams) -> Tuple[List[int], np.ndarray]:
    num_points = len(points)
    temperature, last_visited_step = init_temperature(num_points)

    visited = [0]
    last_visited_step[0] = 0
    update_temperature(0, points, temperature, params)

    for step in range(1, num_points):
        current = visited[-1]

        candidates = [
            i for i in range(num_points)
            if i not in visited and is_valid(i, step, temperature, last_visited_step, params)
        ]

        # Fallback: if all points are blocked, allow the remaining unvisited points.
        if not candidates:
            candidates = [i for i in range(num_points) if i not in visited]

        next_point = min(
            candidates,
            key=lambda i: hybrid_cost(current, i, points, temperature, params),
        )

        visited.append(next_point)
        last_visited_step[next_point] = step
        update_temperature(next_point, points, temperature, params)

    return visited, temperature


def run_baseline_greedy(points: np.ndarray, params: ThermalParams) -> List[int]:
    num_points = len(points)
    visited = [0]

    for _ in range(1, num_points):
        current = visited[-1]
        candidates = [i for i in range(num_points) if i not in visited]
        next_point = min(candidates, key=lambda i: distance_only_cost(current, i, points))
        visited.append(next_point)

    return visited


# =========================================================
# 5) METRICS AND OBP-LIKE OUTPUT
# =========================================================
# Function: quantify performance and export machine-like commands.
# Use: provide a submission-friendly output that can be discussed in an interview.
# Improvement: turns the path into a readable command sequence.


def path_length(points: np.ndarray, path: List[int]) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(np.linalg.norm(points[path[i]] - points[path[i - 1]]) for i in range(1, len(path))))


def dwell_time(temp: float) -> float:
    return 1.0 + 0.5 * float(temp)


def generate_obp(path: List[int], points: np.ndarray, temperature: np.ndarray) -> List[str]:
    commands = []
    for i in path:
        x, y = points[i]
        t = dwell_time(temperature[i])
        commands.append(f"MOVE {x:.2f} {y:.2f}")
        commands.append(f"EXPOSE {t:.2f}")
    return commands


def summarize_path(points: np.ndarray, path: List[int], temperature: np.ndarray | None = None) -> dict:
    summary = {
        "Path length": path_length(points, path),
        "Number of points": len(path),
    }
    if temperature is not None:
        summary["Mean temperature"] = float(np.mean(temperature))
        summary["Max temperature"] = float(np.max(temperature))
        summary["Hot points (> threshold)"] = int(np.sum(temperature > 2.5))
    return summary


# =========================================================
# 6) DATA GENERATION
# =========================================================
# Function: create a reproducible point cloud.
# Use: represent scan locations / melt spots.


def generate_points(num_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((num_points, 2)) * 100.0


# =========================================================
# 7) VISUALIZATION
# =========================================================
# Function: show the scan route and final thermal field.
# Use: help explain the model clearly in the interview.
# Improvement: the app now presents actual figures instead of only text charts.


def plot_path(points: np.ndarray, path: List[int], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    ordered = points[path]
    ax.scatter(points[:, 0], points[:, 1], s=25)
    ax.plot(ordered[:, 0], ordered[:, 1], linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    return fig


def plot_temperature(points: np.ndarray, temperature: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=temperature, s=40)
    fig.colorbar(scatter, ax=ax, label="Temperature")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    return fig


# =========================================================
# 8) STREAMLIT APP
# =========================================================
# This is the submission demo.
# It keeps the interface simple and avoids duplicate widget IDs by using keys.


def main() -> None:
    st.set_page_config(page_title="Thermal-Aware Scan Strategy", layout="wide")

    st.title("🔥 Thermal-Aware Spot-Based Scan Strategy for E-PBF")
    st.caption("Streamlit demo for an open-architecture, Freemelt-style scan planning concept")

    st.sidebar.header("Control Panel")
    num_points = st.sidebar.slider("Number of scan points", 10, 200, 50, key="num_points")
    seed = st.sidebar.number_input("Random seed", value=42, key="seed")

    st.sidebar.subheader("Thermal parameters")
    cooling_radius = st.sidebar.slider("Cooling radius", 5.0, 40.0, 15.0, key="cooling_radius")
    cooldown_steps = st.sidebar.slider("Cooldown steps", 1, 20, 5, key="cooldown_steps")
    heat_increase = st.sidebar.slider("Heat increase", 0.1, 3.0, 1.0, key="heat_increase")
    decay = st.sidebar.slider("Decay", 0.50, 0.99, 0.85, key="decay")
    temp_threshold = st.sidebar.slider("Temperature threshold", 0.5, 5.0, 2.5, key="temp_threshold")
    beta = st.sidebar.slider("Thermal weight", 0.0, 5.0, 2.0, key="beta")

    params = ThermalParams(
        cooling_radius=cooling_radius,
        cooldown_steps=int(cooldown_steps),
        heat_increase=heat_increase,
        decay=decay,
        temp_threshold=temp_threshold,
        alpha=1.0,
        beta=beta,
    )

    points = generate_points(num_points, int(seed))

    thermal_path, final_temperature = run_thermal_optimizer(points, params)
    baseline_path = run_baseline_greedy(points, params)

    baseline_length = path_length(points, baseline_path)
    thermal_length = path_length(points, thermal_path)

    commands = generate_obp(thermal_path, points, final_temperature)
    command_text = "\n".join(commands)

    st.subheader("Key results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Thermal path length", f"{thermal_length:.2f}")
    c2.metric("Baseline path length", f"{baseline_length:.2f}")
    c3.metric("Mean final temperature", f"{np.mean(final_temperature):.2f}")
    c4.metric("Hot points", f"{int(np.sum(final_temperature > params.temp_threshold))}")

    comparison = pd.DataFrame(
        {
            "Method": ["Baseline greedy", "Thermal-aware"],
            "Path length": [baseline_length, thermal_length],
            "Mean temperature": [np.nan, float(np.mean(final_temperature))],
            "Max temperature": [np.nan, float(np.max(final_temperature))],
        }
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Optimized scan path")
        st.pyplot(plot_path(points, thermal_path, "Thermal-aware scan path"), clear_figure=True)

    with right:
        st.subheader("Final thermal field")
        st.pyplot(plot_temperature(points, final_temperature, "Temperature distribution"), clear_figure=True)

    st.subheader("OBP-style output")
    st.code(command_text, language="text")
    st.download_button(
        "Download OBP commands",
        data=command_text,
        file_name="obp_commands.txt",
        mime="text/plain",
        key="download_obp",
    )

    with st.expander("Explain the code section by section"):
        st.markdown(
            """
**Parameters** define how heat spreads and how strongly temperature affects the next scan decision.

**Thermal model** stores temperature and cooldown history, so the scan path reacts to previous exposures.

**Cost function** combines distance and thermal penalty to choose a safer next point.

**Optimizer** builds the scan route and updates the heat field after every exposure.

**OBP output** converts the path into machine-like MOVE / EXPOSE commands.

**Streamlit layer** makes the model easy to run, inspect, and present in the interview.
"""
        )

    with st.expander("Raw data"):
        st.write("Points")
        st.write(points)
        st.write("Temperature")
        st.write(final_temperature)
        st.write("Thermal path")
        st.write(thermal_path)

    st.subheader("Interview talking points")
    st.info(
        """
        This demo shows a spot-based scan strategy with thermal awareness.
        It is a simplified model of how scan sequencing can influence heat accumulation,
        which is relevant to E-PBF systems such as Freemelt ONE.
        """
    )


if __name__ == "__main__":
    main()
