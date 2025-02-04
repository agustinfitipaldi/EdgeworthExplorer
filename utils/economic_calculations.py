import numpy as np
from sympy import symbols, sympify, lambdify
from typing import Callable, List, Tuple

def parse_utility_function(func_str: str) -> Callable:
    """Parse a utility function string into a callable function."""
    x, y = symbols('x y')
    try:
        expr = sympify(func_str)
        return lambdify((x, y), expr, 'numpy')
    except Exception as e:
        raise ValueError(f"Invalid utility function: {str(e)}")

def calculate_mrs(util_func: Callable, x: float, y: float, epsilon: float = 1e-6) -> float:
    """Calculate the Marginal Rate of Substitution at a point."""
    dx = (util_func(x + epsilon, y) - util_func(x, y)) / epsilon
    dy = (util_func(x, y + epsilon) - util_func(x, y)) / epsilon
    return -dx/dy if dy != 0 else np.inf

def calculate_indifference_curves(
    util_func: Callable,
    max_x: float,
    max_y: float,
    num_curves: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Calculate indifference curves for a utility function."""
    x = np.linspace(0.1, max_x, 100)
    curves = []
    
    # Calculate utility levels for different curves
    u_levels = np.linspace(
        util_func(max_x/4, max_y/4),
        util_func(max_x*3/4, max_y*3/4),
        num_curves
    )
    
    for u in u_levels:
        y = []
        for xi in x:
            # Binary search for y value
            low, high = 0.1, max_y
            while high - low > 1e-6:
                mid = (low + high) / 2
                if util_func(xi, mid) < u:
                    low = mid
                else:
                    high = mid
            y.append(low)
        curves.append((x, np.array(y)))
    
    return curves

def calculate_offer_curves(
    util_func_a: Callable,
    util_func_b: Callable,
    total_x: float,
    total_y: float,
    endow_ax: float,
    endow_ay: float,
    endow_bx: float,
    endow_by: float,
    num_points: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate offer curves and contract curve."""
    x_points = []
    y_points = []
    
    # Calculate points along the contract curve
    for t in np.linspace(0, 1, num_points):
        x = total_x * t
        y = total_y * t
        
        # Find point where MRS are equal
        mrs_a = calculate_mrs(util_func_a, x, y)
        mrs_b = calculate_mrs(util_func_b, total_x - x, total_y - y)
        
        if abs(mrs_a - mrs_b) < 1e-3:
            x_points.append(x)
            y_points.append(y)
    
    return np.array(x_points), np.array(y_points)
