import numpy as np

def backtracking_line_search(f, f_grad, x, d, alpha, p=0.5, beta=1e-4):
    y, g = f(x), f_grad(x)
    while f(x + alpha * d) > y + beta * alpha * np.dot(g, d):
        alpha *= p
    x_new = x + alpha * d
    return alpha

def backtracking_line_search_two(f, f_grad, x, d, alpha, p=0.5, beta=1e-4, sigma=0.9):
    y, g = f(x), f_grad(x)
    
    # Adjust the maximum step size, alpha, until it meets the first Wolfe condition
    while f(x + alpha * d) > y + beta * alpha * np.dot(g, d):
        alpha *= p
        
    x_new = x + alpha * d
    # Check if the adjusted design point satisfies the second Wolfe condition
    while np.dot(d, f_grad(x_new)) < sigma * np.dot(d, f_grad(x)):
        alpha *= p
        x_new = x + alpha * d
    return alpha


def strong_backtracking(f, f_grad, x, d, alpha=1, beta=1e-4, sigma=0.1):
    y0, g0, y_prev, alpha_prev = f(x), np.dot(f_grad(x), d), np.nan, 0
    alpha_lo, alpha_hi = np.nan, np.nan
    
    # Bracket phase
    while True:
        y = f(x + alpha * d)
        if y > y0 + beta * alpha * g0 or (np.isnan(y_prev) & (y >= y_prev)):
            alpha_lo, alpha_hi = alpha_prev, alpha
            break
        g = np.dot(d, f_grad(x + alpha * d))
        if np.abs(g) <= -sigma * g0:
            return alpha
        elif g >= 0:
            alpha_lo, alpha_hi = alpha, alpha_prev
            break
        y_prev, alpha_prev, alpha = y, alpha, 2 * alpha
        
    # Zoom phase
    y_lo = f(x + alpha_lo * d)
    while True:
        alpha = (alpha_lo + alpha_hi) / 2
        y = f(x + alpha * d)
        if y > y0 + beta * alpha * g0 or y >= y_lo:
            alpha_hi = alpha
        else:
            g = np.dot(d, f_grad(x + alpha * d))
            if np.abs(g) <= -sigma * g0:
                return alpha
            elif g * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha
