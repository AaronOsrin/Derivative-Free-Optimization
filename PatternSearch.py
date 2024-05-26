import numpy as np

class PatternSearch:
    def __init__(self, func, x0, gamma0=1.0, gamma_tol=1e-6, theta_max=0.5, rho=lambda t: t/2, phi=2, max_iter=1000):
        self.func = func
        self.x0 = np.array(x0, dtype=float)
        self.gamma0 = gamma0
        self.gamma_tol = gamma_tol
        self.theta_max = theta_max
        self.rho = rho
        self.phi = phi
        self.max_iter = max_iter

    def search(self):
        x = self.x0.copy()
        gamma = self.gamma0
        path = [x.copy()]
        function_values = [self.func(x)]
        for _ in range(self.max_iter):
            if gamma <= self.gamma_tol:
                break

            found_better = False
            for direction in self._get_directions(len(x)):
                new_x = x + gamma * direction
                if self.func(new_x) < self.func(x) - self.rho(gamma):
                    x = new_x
                    gamma *= self.phi
                    found_better = True
                    break

            if not found_better:
                gamma *= self.theta_max

            path.append(x.copy())
            function_values.append(self.func(x))

        return x, self.func(x), path, function_values

    def _get_directions(self, n):
        directions = []
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1
            directions.append(e)
            directions.append(-e)
        return directions
