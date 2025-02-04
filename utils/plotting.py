import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple

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
            line=dict(color='blue', width=1, opacity=0.5),
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
            line=dict(color='red', width=1, opacity=0.5),
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