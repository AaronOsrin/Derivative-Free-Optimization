import numpy as np

class GradientDescent:
    def __init__(self, func, grad, x0, alpha=0.01, tol=1e-6, max_iter=1000):
        self.func = func
        self.grad = grad
        self.x0 = np.array(x0, dtype=float)
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

    def search(self):
        x = self.x0.copy()
        path = [x.copy()]
        function_values = [self.func(x)]
        for _ in range(self.max_iter):
            grad = self.grad(x)
            new_x = x - self.alpha * grad
            if np.linalg.norm(new_x - x) < self.tol:
                break
            x = new_x
            path.append(x.copy())
            function_values.append(self.func(x))
        return x, self.func(x), path, function_values
