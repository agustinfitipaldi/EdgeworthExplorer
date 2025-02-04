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

    u_levels = np.linspace(
        util_func(max_x/4, max_y/4),
        util_func(max_x*3/4, max_y*3/4),
        num_curves
    )

    for u in u_levels:
        y = []
        for xi in x:
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

def find_walrasian_equilibrium(
    util_func_a: Callable,
    util_func_b: Callable,
    total_x: float,
    total_y: float,
    num_prices: int = 50
) -> List[Tuple[float, float]]:
    """Find Walrasian equilibrium points by checking different price ratios."""
    equilibrium_points = []

    for i in range(num_prices):
        price_ratio = np.exp(np.linspace(-3, 3, num_prices)[i])

        def demand_excess(x):
            # Find y through price ratio
            y = total_y - (price_ratio * (total_x - x))
            if y < 0 or y > total_y:
                return float('inf')

            mrs_a = calculate_mrs(util_func_a, x, y)
            mrs_b = calculate_mrs(util_func_b, total_x - x, total_y - y)

            return abs(mrs_a - price_ratio) + abs(mrs_b - price_ratio)

        # Binary search for equilibrium
        left, right = 0.1, total_x - 0.1
        for _ in range(20):
            mid = (left + right) / 2
            if demand_excess(mid - 1e-5) > demand_excess(mid + 1e-5):
                left = mid
            else:
                right = mid

        x_eq = (left + right) / 2
        y_eq = total_y - (price_ratio * (total_x - x_eq))

        if 0 < x_eq < total_x and 0 < y_eq < total_y:
            if demand_excess(x_eq) < 0.1:  # Tolerance for equilibrium
                equilibrium_points.append((x_eq, y_eq))

    return equilibrium_points

def calculate_offer_curves(
    util_func_a: Callable,
    util_func_b: Callable,
    total_x: float,
    total_y: float,
    endow_ax: float,
    endow_ay: float,
    endow_bx: float,
    endow_by: float,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate contract curve points."""
    x_points = []
    y_points = []

    # Calculate points along the contract curve
    for t in np.linspace(0.01, 0.99, num_points):
        x = total_x * t
        y = total_y * t

        # Try different y values for each x
        for s in np.linspace(0.01, 0.99, 20):
            y_test = total_y * s

            mrs_a = calculate_mrs(util_func_a, x, y_test)
            mrs_b = calculate_mrs(util_func_b, total_x - x, total_y - y_test)

            if abs(mrs_a - mrs_b) < 0.1:  # Relaxed tolerance
                x_points.append(x)
                y_points.append(y_test)
                break

    return np.array(x_points), np.array(y_points)