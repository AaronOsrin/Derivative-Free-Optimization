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
    
    def ackley_grad(self, x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(x)
        sum1 = np.sum(x**2)
        grad1 = a * b * np.exp(-b * np.sqrt(sum1 / d)) * (x / (d * np.sqrt(sum1 / d)))
        grad2 = np.exp(np.sum(np.cos(c * x)) / d) * (c / d) * np.sin(c * x)
        return grad1 + grad2

    def quadratic(self, x):
        Q = np.eye(len(x))
        return 0.5 * np.dot(x, Q.dot(x))

    def abs_function(self, x):
        return (np.abs(x[0])+np.abs(x[1]))
    
    def quadratic_grad(self, x):
        Q = np.eye(len(x))
        return Q.dot(x)
    
    def rosenbrock(self, x):
        return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    def rosenbrock_grad(self, x):
        grad = np.zeros_like(x)
        grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        grad[1] = 200 * (x[1] - x[0]**2)
        return grad
    
    def sphere(self, x):
        return sum(x**2)
    
    def sphere_grad(self, x):
        return 2 * x



    def beale(self, x):
        return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2

    def beale_grad(self,x):
        f1 = 1.5 - x[0]+ x[0]*x[1]
        f2 = 2.25 - x[0] + x[0]*x[1]**2
        f3 = 2.625 - x[0] + x[0]*x[1]**3

        df1_dx = -1 + x[1]
        df2_dx = -1 + x[1]**2
        df3_dx = -1 + x[1]**3

        df_dx = 2 * f1 * df1_dx + 2 * f2 * df2_dx + 2 * f3 * df3_dx

        # Partial derivative with respect to y
        df1_dy = x[0]
        df2_dy = 2 * x[0] * x[1]
        df3_dy = 3 * x[0] * x[1]**2

        df_dy = 2 * f1 * df1_dy + 2 * f2 * df2_dy + 2 * f3 * df3_dy

        return np.array([df_dx, df_dy])
        

    def booth(self, x):
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

    def matyas(self, x):
        return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

    def himmelblau(self, x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    def levi(self, x):
        return np.sin(3 * np.pi * x[0])**2 + (x[0] - 1)**2 * (1 + np.sin(3 * np.pi * x[1])**2) + (x[1] - 1)**2 * (1 + np.sin(2 * np.pi * x[1])**2)

    def three_hump_camel(self, x):
        return 2 * x[0]**2 - 1.05 * x[0]**4 + (x[0]**6) / 6 + x[0] * x[1] + x[1]**2

    def rastrigin(self, x):
        A = 10
        return A * len(x) + sum(x**2 - A * np.cos(2 * np.pi * x))

    def griewank(self, x):
        return 1 + sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

    def schaffer_n2(self, x):
        return 0.5 + (np.sin(x[0]**2 - x[1]**2)**2 - 0.5) / (1 + 0.001 * (x[0]**2 + x[1]**2))**2
