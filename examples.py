import numpy as np

class Examples:
    def ackley(self, x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
        return term1 + term2 + a + np.exp(1)
    
    def quadratic(self, x):
        Q = np.eye(len(x))
        return 0.5 * np.dot(x, Q.dot(x))
    
    def rosenbrock(self, x):
        return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    
    def sphere(self, x):
        return sum(x**2)
