import unittest
import numpy as np
from PatternSearch import PatternSearch
from examples import Examples
from utils_opt import plot_contour_with_paths, plot_function_values

class TestPatternSearch(unittest.TestCase):

    def setUp(self):
        self.examples = Examples()
        self.x0 = np.array([1.0, 1.0])
        self.gamma0 = 1.0
        self.gamma_tol = 1e-6
        self.theta_max = 0.5
        self.rho = lambda t: t / 2
        self.phi = 2
        self.max_iter = 1000

    def run_pattern_search(self, func, name):
        psm = PatternSearch(func, self.x0, self.gamma0, self.gamma_tol, self.theta_max, self.rho, self.phi, self.max_iter)
        x_min, f_min, path, function_values = psm.search()
        print(f"{name} - Minimum: {x_min}, Function value: {f_min}")

        # Plotting
        plot_contour_with_paths(func, [-2, 2, -2, 2], paths=[(path, "Pattern Search")])
        plot_function_values((function_values, "Pattern Search"))

    def test_ackley(self):
        self.run_pattern_search(self.examples.ackley, "Ackley")

    def test_quadratic(self):
        self.run_pattern_search(self.examples.quadratic, "Quadratic")

    def test_rosenbrock(self):
        self.run_pattern_search(self.examples.rosenbrock, "Rosenbrock")

    def test_sphere(self):
        self.run_pattern_search(self.examples.sphere, "Sphere")

if __name__ == '__main__':
    unittest.main()
