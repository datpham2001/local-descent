from utils import*
def rosenbrock(x, b=5):   # hàm rosenbrock
    return (x[0] - 1)**2 + b * (x[1] - x[0]**2)**2

def rosenbrock_grad(x, b=5):  #đạo hàm bậc 1 (gradient)
    return np.array([2 * (x[0] - 1) - 4 * b * (x[1] - x[0]**2) * x[0], 2 * b * (x[1] - x[0]**2)])

def rosenbrock_hess(x, b=5): # đạo hầm bậc 2 (Hessian)
    return np.array([[2 - 4 * b * x[1] + 12 * b * x[0]**2, -4 * b * x[0]]\
                     ,[-4 * b * x[0], 2 * b]])
