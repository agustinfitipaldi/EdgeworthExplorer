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
    num_prices: int = 100
) -> List[Tuple[float, float]]:
    """Find Walrasian equilibrium points by checking different price ratios."""
    equilibrium_points = []
    price_ratios = np.exp(np.linspace(-5, 5, num_prices))  # Wider range

    for price_ratio in price_ratios:
        def demand_excess(x):
            if x <= 0.1 or x >= total_x - 0.1:
                return float('inf')

            # Try different y values for this x
            best_excess = float('inf')
            best_y = None

            for t in np.linspace(0.1, 0.9, 20):
                y_test = total_y * t

                if 0.1 < y_test < total_y - 0.1:
                    mrs_a = calculate_mrs(util_func_a, x, y_test)
                    mrs_b = calculate_mrs(util_func_b, total_x - x, total_y - y_test)

                    excess = abs(mrs_a - price_ratio) + abs(mrs_b - price_ratio)
                    if excess < best_excess:
                        best_excess = excess
                        best_y = y_test

            return best_excess if best_y is not None else float('inf')

        # Grid search for initial point
        x_grid = np.linspace(0.1, total_x - 0.1, 30)
        x_candidates = [(x, demand_excess(x)) for x in x_grid]
        x_candidates.sort(key=lambda x: x[1])

        # Take the best candidates and optimize them
        for x_init, excess in x_candidates[:3]:
            if excess < 1.0:  # Relaxed initial tolerance
                # Local optimization
                x_current = x_init
                step = 0.1

                while step > 1e-4:
                    improved = False
                    for x_new in [x_current - step, x_current + step]:
                        if 0.1 < x_new < total_x - 0.1:
                            if demand_excess(x_new) < demand_excess(x_current):
                                x_current = x_new
                                improved = True
                                break
                    if not improved:
                        step *= 0.5

                # Find corresponding y value
                best_y = None
                best_excess = float('inf')

                for t in np.linspace(0.1, 0.9, 50):
                    y_test = total_y * t
                    if 0.1 < y_test < total_y - 0.1:
                        mrs_a = calculate_mrs(util_func_a, x_current, y_test)
                        mrs_b = calculate_mrs(util_func_b, total_x - x_current, total_y - y_test)
                        excess = abs(mrs_a - price_ratio) + abs(mrs_b - price_ratio)

                        if excess < best_excess:
                            best_excess = excess
                            best_y = y_test

                if best_y is not None and best_excess < 1.0:  # Relaxed final tolerance
                    equilibrium_points.append((x_current, best_y))

    # Remove nearby points
    filtered_points = []
    for p1 in equilibrium_points:
        if not any(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < 1.0
                  for p2 in filtered_points):
            filtered_points.append(p1)

    return filtered_points

def calculate_offer_curves(
    util_func_a: Callable,
    util_func_b: Callable,
    total_x: float,
    total_y: float,
    endow_ax: float,
    endow_ay: float,
    endow_bx: float,
    endow_by: float,
    num_points: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate contract curve points using improved optimization."""
    x_points = []
    y_points = []

    # Use more sophisticated sampling
    for t in np.linspace(0.05, 0.95, num_points):
        x_base = total_x * t
        y_values = np.linspace(0.05 * total_y, 0.95 * total_y, 50)

        best_diff = float('inf')
        best_y = None

        for y_test in y_values:
            mrs_a = calculate_mrs(util_func_a, x_base, y_test)
            mrs_b = calculate_mrs(util_func_b, total_x - x_base, total_y - y_test)

            diff = abs(mrs_a - mrs_b)
            if diff < best_diff:
                best_diff = diff
                best_y = y_test

        # Local optimization around best point
        if best_y is not None and best_diff < 1.0:  # Relaxed initial tolerance
            dy = 0.1 * total_y
            while dy > 1e-6 * total_y:
                improved = False
                for y_new in [best_y - dy, best_y + dy]:
                    if 0.01 * total_y < y_new < 0.99 * total_y:
                        mrs_a = calculate_mrs(util_func_a, x_base, y_new)
                        mrs_b = calculate_mrs(util_func_b, total_x - x_base, total_y - y_new)
                        diff = abs(mrs_a - mrs_b)

                        if diff < best_diff:
                            best_diff = diff
                            best_y = y_new
                            improved = True

                if not improved:
                    dy *= 0.5

            if best_diff < 0.1:  # Final tolerance
                x_points.append(x_base)
                y_points.append(best_y)

    # Smooth the curves using moving average
    if len(x_points) > 5:
        window = 5
        y_smooth = np.convolve(y_points, np.ones(window)/window, mode='valid')
        x_smooth = x_points[window-1:][:len(y_smooth)]
        return np.array(x_smooth), np.array(y_smooth)

    return np.array(x_points), np.array(y_points)