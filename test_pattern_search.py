import unittest
import numpy as np
from PatternSearch import PatternSearch
from GradientDescent import GradientDescent
from examples import Examples
from utils_opt import plot_contour_with_paths, plot_function_values

class TestPatternSearch(unittest.TestCase):

    def setUp(self):
        self.examples = Examples()
        self.x0 = np.array([1.5, 1.5])
        self.gamma0 = 1.0
        self.gamma_tol = 1e-6
        self.theta_max = 0.5
        self.rho = lambda t: t / 2
        self.phi = 2
        self.max_iter = 1000
        self.alpha = 0.01
        self.tol = 1e-6

    def run_pattern_search(self, func, grad, name):
        psm = PatternSearch(func, self.x0, self.gamma0, self.gamma_tol, self.theta_max, self.rho, self.phi, self.max_iter)
        x_min_psm, f_min_psm, path_psm, function_values_psm = psm.search()
        
        if grad is not None:
            gd = GradientDescent(func, grad, self.x0, self.alpha, self.tol, self.max_iter)
            x_min_gd, f_min_gd, path_gd, function_values_gd = gd.search()
            print(f"{name} - Pattern Search Minimum: {x_min_psm}, Function value: {f_min_psm}")
            print(f"{name} - Gradient Descent Minimum: {x_min_gd}, Function value: {f_min_gd}")
            plot_contour_with_paths(func, [-4, 4, -2, 2], name, paths=[(path_psm, "Pattern Search"), (path_gd, "Gradient Descent")])
            plot_function_values(name, (function_values_psm, "Pattern Search"), (function_values_gd, "Gradient Descent"))
        else:
            print(f"{name} - Pattern Search Minimum: {x_min_psm}, Function value: {f_min_psm}")
            plot_contour_with_paths(func, [-2, 2, -2, 2], name, paths=[(path_psm, "Pattern Search")])
            plot_function_values(name, (function_values_psm, "Pattern Search"))

    #def test_ackley(self):
    #    self.run_pattern_search(self.examples.ackley, self.examples.ackley_grad, "Ackley")

    #def test_quadratic(self):
    #    self.run_pattern_search(self.examples.quadratic, self.examples.quadratic_grad, "Quadratic")

   #def test_rosenbrock(self):
    #    self.run_pattern_search(self.examples.rosenbrock, self.examples.rosenbrock_grad, "Rosenbrock")

    def test_sphere(self):
        self.run_pattern_search(self.examples.sphere, self.examples.sphere_grad, "Sphere")

    # For functions without gradients, we'll pass None
    def test_beale(self):
        self.run_pattern_search(self.examples.beale, self.examples.beale_grad, "Beale")

    #def test_booth(self):
    #    self.run_pattern_search(self.examples.booth, None, "Booth")

    #def test_matyas(self):
    #    self.run_pattern_search(self.examples.matyas, None, "Matyas")

    #def test_himmelblau(self):
    #    self.run_pattern_search(self.examples.himmelblau, None, "Himmelblau")

    #def test_levi(self):
    #    self.run_pattern_search(self.examples.levi, None, "Levi")

    #def test_three_hump_camel(self):
    #    self.run_pattern_search(self.examples.three_hump_camel, None, "Three-Hump Camel")

    #def test_rastrigin(self):
    #    self.run_pattern_search(self.examples.rastrigin, None, "Rastrigin")

    #def test_griewank(self):
     #   self.run_pattern_search(self.examples.griewank, None, "Griewank")

    #def test_schaffer_n2(self):
    #    self.run_pattern_search(self.examples.schaffer_n2, None, "Schaffer N. 2")
    
    def test_abs_function(self):
        self.run_pattern_search(self.examples.abs_function, None, "Absolute Value")

if __name__ == '__main__':
    unittest.main()
