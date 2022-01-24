import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy import linspace

#Van der Pol ODE solution for mu = 1
def VDP(t,z):
    x,y = z[0], z[1]
    dz = (y, mu*(1-x**2)*y - x)
    return (dz)
mu = 1
ti = 0
tf = 20
tot= linspace(ti,tf, 10000)
ipts = [[1,0],[-1,0],[0,1,],[-2,1],[2,1],[1,2],[-0.01,0],[0.01,1],[0.01,0.01]]
for ipt in ipts:
    sol = solve_ivp(VDP, [ti,tf], ipt, t_eval=tot)
    plt.plot(sol.y[0], sol.y[1], '-')
plt.grid()
plt.legend([f"$x_0, y_0 = {ipt[0], ipt[1]}$" for ipt in ipts],loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel("x")
plt.ylabel("y")

#Van der Pol ODE for mu = 10
def VDP(t,z):
    x,y = z[0], z[1]
    dz = (y, mu*(1-x**2)*y - x)
    return (dz)
mu = 10
ti = 0
tf = 20
tot= linspace(ti,tf, 10000)
ipts = [[1,0],[-1,0],[0,1,],[-2,1],[2,1],[1,2],[-0.01,0],[0.01,1],[0.01,0.01]]
for ipt in ipts:
    sol = solve_ivp(VDP, [ti,tf], ipt, t_eval=tot)
    plt.plot(sol.y[0], sol.y[1], '-')
plt.grid()
plt.legend([f"$x_0, y_0 = {ipt[0], ipt[1]}$" for ipt in ipts],loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel("x")
plt.ylabel("y")

#Period of the Van der Pol ODE 
u= linspace(1,11, 10)
T = (3-2*np.log(2))*u + 3*(2.2338/(u**(1/3)))
plt.grid()
plt.plot(u,T)
plt.xlabel("u")
plt.ylabel("T(u)")




