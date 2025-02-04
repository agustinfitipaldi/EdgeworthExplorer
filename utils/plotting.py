import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple, Callable

def create_edgeworth_box(
    ic_a: List[Tuple[np.ndarray, np.ndarray]],
    ic_b: List[Tuple[np.ndarray, np.ndarray]],
    offer_curves: Tuple[np.ndarray, np.ndarray],
    equilibrium_points: List[Tuple[float, float]],
    total_x: float,
    total_y: float,
    endow_ax: float,
    endow_ay: float
) -> go.Figure:
    """Create the Edgeworth box visualization."""
    fig = go.Figure()

    # Add indifference curves for Agent A
    for x, y in ic_a:
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(color='rgba(0,0,255,0.3)', width=1),
            name="Agent A's Indifference Curve",
            hoverinfo='skip',
            showlegend=False
        ))

    # Add indifference curves for Agent B (inverted)
    for x, y in ic_b:
        fig.add_trace(go.Scatter(
            x=total_x - x,
            y=total_y - y,
            mode='lines',
            line=dict(color='rgba(255,0,0,0.3)', width=1),
            name="Agent B's Indifference Curve",
            hoverinfo='skip',
            showlegend=False
        ))

    # Add contract curve with increased visibility
    fig.add_trace(go.Scatter(
        x=offer_curves[0],
        y=offer_curves[1],
        mode='lines',
        line=dict(color='green', width=3),
        name='Contract Curve'
    ))

    # Add Walrasian equilibrium points
    if equilibrium_points:
        x_eq = [p[0] for p in equilibrium_points]
        y_eq = [p[1] for p in equilibrium_points]
        fig.add_trace(go.Scatter(
            x=x_eq,
            y=y_eq,
            mode='markers',
            marker=dict(
                symbol='circle',
                size=12,
                color='purple',
                line=dict(color='white', width=1)
            ),
            name='Walrasian Equilibrium'
        ))

    # Add initial endowment point
    fig.add_trace(go.Scatter(
        x=[endow_ax],
        y=[endow_ay],
        mode='markers',
        marker=dict(symbol='star', size=15, color='gold'),
        name='Initial Endowment'
    ))

    # Configure layout
    fig.update_layout(
        title='Edgeworth Box Diagram',
        xaxis=dict(
            title='Good X',
            range=[0, total_x],
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Good Y',
            range=[0, total_y],
            gridcolor='lightgray'
        ),
        showlegend=True,
        plot_bgcolor='white',
        width=800,
        height=800
    )

    # Add box borders
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=total_x, y1=total_y,
        line=dict(color="black", width=2),
        fillcolor="rgba(0,0,0,0)"
    )

    return fig

def create_utility_surfaces(
    util_func_a: Callable,
    util_func_b: Callable,
    total_x: float,
    total_y: float,
    num_points: int = 50
) -> go.Figure:
    """Create 3D visualization of utility surfaces."""
    # Create grid of points
    x = np.linspace(0.1, total_x-0.1, num_points)
    y = np.linspace(0.1, total_y-0.1, num_points)
    X, Y = np.meshgrid(x, y)

    # Calculate utility values for both agents
    Z_a = np.zeros_like(X)
    Z_b = np.zeros_like(X)

    for i in range(num_points):
        for j in range(num_points):
            x_val = X[i,j]
            y_val = Y[i,j]
            Z_a[i,j] = util_func_a(x_val, y_val)
            Z_b[i,j] = util_func_b(total_x - x_val, total_y - y_val)

    # Create 3D figure
    fig = go.Figure()

    # Add surface for Agent A
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z_a,
        colorscale='Blues',
        name="Agent A's Utility",
        showscale=False,
        opacity=0.8
    ))

    # Add surface for Agent B
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z_b,
        colorscale='Reds',
        name="Agent B's Utility",
        showscale=False,
        opacity=0.8
    ))

    # Configure 3D layout
    fig.update_layout(
        title='3D Utility Surfaces',
        scene=dict(
            xaxis_title='Good X',
            yaxis_title='Good Y',
            zaxis_title='Utility Level',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=800,
        height=800,
        showlegend=True
    )

    return fig