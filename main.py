import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.economic_calculations import (
    parse_utility_function, calculate_indifference_curves,
    calculate_offer_curves, find_walrasian_equilibrium
)
from utils.plotting import create_edgeworth_box, create_utility_surfaces

st.set_page_config(page_title="Edgeworth Box Visualization", layout="wide")

st.title("Interactive Edgeworth Box Visualization")

st.markdown("""
This tool helps visualize the Edgeworth box diagram with indifference curves, contract curve, and Walrasian equilibrium points.
Enter the utility functions and initial endowments below.
""")

# Input columns for utility functions and endowments
col1, col2 = st.columns(2)

with col1:
    st.subheader("Agent A")
    utility_a = st.text_input(
        "Utility Function A",
        value="x**0.5 * y**0.5",
        help="Enter utility function using x and y as variables (e.g., x**0.5 * y**0.5)"
    )
    endow_ax = st.number_input("Initial Endowment X for A", value=10.0, min_value=0.1)
    endow_ay = st.number_input("Initial Endowment Y for A", value=5.0, min_value=0.1)

with col2:
    st.subheader("Agent B")
    utility_b = st.text_input(
        "Utility Function B",
        value="x**0.3 * y**0.7",
        help="Enter utility function using x and y as variables (e.g., x**0.3 * y**0.7)"
    )
    endow_bx = st.number_input("Initial Endowment X for B", value=5.0, min_value=0.1)
    endow_by = st.number_input("Initial Endowment Y for B", value=10.0, min_value=0.1)

# Parse utility functions and calculate curves
try:
    util_func_a = parse_utility_function(utility_a)
    util_func_b = parse_utility_function(utility_b)

    total_x = endow_ax + endow_bx
    total_y = endow_ay + endow_by

    # Calculate curves and equilibrium points
    ic_a = calculate_indifference_curves(util_func_a, total_x, total_y)
    ic_b = calculate_indifference_curves(util_func_b, total_x, total_y)

    offer_curves = calculate_offer_curves(
        util_func_a, util_func_b,
        total_x, total_y,
        endow_ax, endow_ay,
        endow_bx, endow_by
    )

    equilibrium_points = find_walrasian_equilibrium(
        util_func_a, util_func_b,
        total_x, total_y
    )

    # Create 2D visualization
    fig_2d = create_edgeworth_box(
        ic_a, ic_b, offer_curves,
        equilibrium_points,
        total_x, total_y,
        endow_ax, endow_ay
    )

    st.plotly_chart(fig_2d, use_container_width=True)

    # Create 3D visualization
    fig_3d = create_utility_surfaces(
        util_func_a, util_func_b,
        total_x, total_y
    )

    st.plotly_chart(fig_3d, use_container_width=True)

except Exception as e:
    st.error(f"Error in calculations: {str(e)}")

# Add explanatory text
st.markdown("""
### How to use this visualization:
1. Enter utility functions for both agents using x and y as variables
2. Set initial endowments for both goods (X and Y) for each agent
3. The visualization will update automatically
4. Hover over the curves and points to see detailed information

### Features:
- Blue curves: Agent A's indifference curves
- Red curves: Agent B's indifference curves
- Green curve: Contract curve (points where marginal rates of substitution are equal)
- Purple dots: Walrasian equilibrium points (where markets clear)
- Gold star: Initial endowment point

### 3D Visualization:
- Blue surface: Agent A's utility function
- Red surface: Agent B's utility function (inverted)
- The intersection of these surfaces helps visualize Pareto-efficient allocations
""")