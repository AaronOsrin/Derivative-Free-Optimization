import unittest
import numpy as np
from PatternSearch import PatternSearch
from GradientDescent import GradientDescent
from examples import Examples
from utils_opt import plot_contour_with_paths, plot_function_values

class TestPatternSearch(unittest.TestCase):

    def setUp(self):
        self.examples = Examples()
        self.gamma0 = 1.0
        self.gamma_tol = 1e-6
        self.theta_max = 0.5
        self.rho = lambda t: t / 2
        self.phi = 2
        self.max_iter = 1000
        self.alpha = 0.01
        self.tol = 1e-6

    def runner(self, func, grad, name,x0, x_plot, y_plot):
        psm = PatternSearch(func, x0, self.gamma0, self.gamma_tol, self.theta_max, self.rho, self.phi, self.max_iter)
        x_min_psm, f_min_psm, path_psm, function_values_psm = psm.search()
        
        if grad is not None:
            gd = GradientDescent(func, grad, x0, self.alpha, self.tol, self.max_iter)
            x_min_gd, f_min_gd, path_gd, function_values_gd = gd.search()
            print(f"{name} - Pattern Search Minimum: {x_min_psm}, Function value: {f_min_psm}")
            print(f"{name} - Gradient Descent Minimum: {x_min_gd}, Function value: {f_min_gd}")
            plot_contour_with_paths(func, [-x_plot, x_plot, -y_plot, y_plot], name, paths=[(path_psm, "Pattern Search"), (path_gd, "Gradient Descent")])
            plot_function_values(name, (function_values_psm, "Pattern Search"), (function_values_gd, "Gradient Descent"))
        else:
            print(f"{name} - Pattern Search Minimum: {x_min_psm}, Function value: {f_min_psm}")
            plot_contour_with_paths(func, [-x_plot, x_plot, -y_plot, y_plot], name, paths=[(path_psm, "Pattern Search")])
            plot_function_values(name, (function_values_psm, "Pattern Search"))


    def test_sphere(self):
        x0 = np.array([1.5,1.5])
        self.runner(self.examples.sphere, self.examples.sphere_grad, "Sphere",x0,  2, 2)

    
    def test_beale(self):
        x0 = np.array([0,0])
        self.runner(self.examples.beale, self.examples.beale_grad, "Beale",x0,  4.5, 2)
    
    def test_abs_function(self):
        x0 = np.array([1.5,1.5])
        self.runner(self.examples.abs_function, None, "Absolute Value",x0,  2, 2)

if __name__ == '__main__':
    unittest.main()
