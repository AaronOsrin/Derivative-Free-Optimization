import numpy as np

class Examples:

    def abs_function(self, x):
        return (np.abs(x[0])+np.abs(x[1]))
    
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

        
        df1_dy = x[0]
        df2_dy = 2 * x[0] * x[1]
        df3_dy = 3 * x[0] * x[1]**2

        df_dy = 2 * f1 * df1_dy + 2 * f2 * df2_dy + 2 * f3 * df3_dy

        return np.array([df_dx, df_dy])