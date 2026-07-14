# =================================================================================================
#  Illustrative bifurcation in the minimal phase space of the effective replicator–niche dynamics.
# =================================================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.integrate import odeint
from scipy.optimize import fsolve

# 1. Model
def biomatter_dynamics(y, t, delta, gamma, alpha, beta, eta, lmbda):
    R, S = y
    dR_dt = -delta * R + beta * S - eta * (R**2)
    dS_dt = alpha * R - gamma * S - lmbda * (S**2)
    return [dR_dt, dS_dt]

# 2. Attractor
def find_attractor(params):
    delta, gamma, alpha, beta, eta, lmbda = params
    K = (alpha * beta) / (delta * gamma)
    def nullclines(vars):
            R, S = vars
            eq1 = -delta * R + beta * S - eta * (R**2)
            eq2 = alpha * R - gamma * S - lmbda * (S**2)
            return [eq1, eq2]
        
    guess_inicial = (1.0, 1.0)
    R_opt, S_opt = fsolve(nullclines, guess_inicial)
        
    return (R_opt, S_opt)

# 3. Phase Space
def plot_phase_space(ax, params, title, max_val):
    delta, gamma, alpha, beta, eta, lmbda = params
    
    # Grid for vector field
    R_grid, S_grid = np.meshgrid(np.linspace(0, max_val, 25), 
                                 np.linspace(0, max_val, 25))
    
    # Derivatives
    dR = -delta * R_grid + beta * S_grid - eta * (R_grid**2)
    dS = alpha * R_grid - gamma * S_grid - lmbda * (S_grid**2)
    
    # Vector field
    ax.streamplot(R_grid, S_grid, dR, dS, color='lightgray', density=1.2, linewidth=1)
    
    # Null isoclines
    R_line = np.linspace(0, max_val, 100)
    S_Rnull = (delta / beta) * R_line + (eta / beta) * (R_line**2)
    S_line = np.linspace(0, max_val, 100)
    R_Snull = (gamma / alpha) * S_line + (lmbda / alpha) * (S_line**2)
    
    ax.plot(R_line, S_Rnull, 'b--', linewidth=1.5, label=r'Isocline $\dot{R} = 0$')
    ax.plot(R_Snull, S_line, 'g--', linewidth=1.5, label=r'Isocline $\dot{S} = 0$')
    
    # Trajectories
    t = np.linspace(0, 50, 1000)
    initial_conditions = [(0.5, 0.52), (max_val*0.8, 0.2), (0.2, max_val*0.8)]
    for idx, y0 in enumerate(initial_conditions):
        sol = odeint(biomatter_dynamics, y0, t, args=params)
        label = 'Trayectories' if idx == 0 else None
        ax.plot(sol[:, 0], sol[:, 1], 'r-', linewidth=1, label=label)
        ax.plot(y0[0], y0[1], 'ko', markersize=5)
        
    cx, cy = find_attractor(params)
    
    cuenca = Circle((cx, cy), radius=max_val*0.12, color='purple', alpha=0.15)
    ax.add_patch(cuenca)
    ax.plot(cx, cy, 'o', color='purple', markersize=8, label=f'Attractor')

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel('Replicator ($R$)', fontsize=12)
    ax.set_ylabel('Niche ($S$)', fontsize=12)
    ax.set_title(title, fontsize=14)

# 4. Parameters
params_OFF = (1.0, 1.0, 0.5, 0.4, 0.1, 0.1) # K = 0.25
params_ON  = (1.0, 1.0, 2.0, 1.9, 0.5, 0.5) # K = 4.0

# 5. Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plot_phase_space(ax1, params_OFF, r'Phase OFF ($K < 1$)', max_val=3.0)
plot_phase_space(ax2, params_ON,  r'Phase ON ($K > 1$)', max_val=3.0)

plt.tight_layout()
plt.savefig('fasesRS.png', dpi=300)