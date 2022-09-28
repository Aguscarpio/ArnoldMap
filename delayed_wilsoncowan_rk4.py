#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022

@author: aguscarpio99
"""

import numpy as np
from numba import njit

@njit()
def rk4_numba(Z_0_vec, dt, N_steps, rox, roy, mu, K, tau_steps):
    dt2=dt/2
    dt6=dt/6
    first = Z_0_vec.size
    Z_OT = np.zeros(first+N_steps, dtype="complex")
    Z_OT[:first] = Z_0_vec
    Z = Z_OT[first-1]
    for i in range(first, N_steps+first):
        Z_orig = Z
        Z_tau = Z_OT[i-tau_steps]

        k1 = mu*(-Z + 1/(1+np.exp(-(rox + 10*Z.real - 10*Z.imag)))
                   + 1j*1/(1+np.exp(-(roy + 10*Z.real + 10*Z.imag + K*Z_tau.real))))

        Z=Z_orig+dt2*k1

        k2 = mu*(-Z + 1/(1+np.exp(-(rox + 10*Z.real - 10*Z.imag)))
                   + 1j*1/(1+np.exp(-(roy + 10*Z.real + 10*Z.imag + K*Z_tau.real))))

        Z=Z_orig+dt2*k2
        k3 = mu*(-Z + 1/(1+np.exp(-(rox + 10*Z.real - 10*Z.imag)))
                   + 1j*1/(1+np.exp(-(roy + 10*Z.real + 10*Z.imag + K*Z_tau.real))))
        Z=Z_orig+dt*k3
        k4 = mu*(-Z + 1/(1+np.exp(-(rox + 10*Z.real - 10*Z.imag)))
                 + 1j*1/(1+np.exp(-(roy + 10*Z.real + 10*Z.imag + K*Z_tau.real))))
        Z=Z_orig+dt6*(2.0*(k2+k3)+k1+k4)
        Z_OT[i] = Z
    return Z_OT[tau_steps:]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    from save import save_data
    from matplotlib.animation import FuncAnimation
    from period_funcs import period
    plt.style.use("seaborn")

    dt=0.00004
    N_steps=200000
    animation_jump = 30

    x_0 = 0.0705
    y_0 = 0.15942
    x_0 = 0.04
    y_0 = 0.25
    Z_0=x_0+1j*y_0

    #  rox = -4.5
    #  roy = -8.4
    rox = -2.3
    roy = -9
    mu = 150
    K = 0
    tau = 0.04880

    tau_steps = int(tau/dt)
    use_last = True
    BACKWARD = False

    if BACKWARD:
        dt = -dt

    data_dir = ""

    if use_last:
        with open(f"{data_dir}last_tau_vector.npy", "rb") as f:
            zvec = np.load(f)
        Z_0_vec = zvec[-tau_steps:]
    else:
        Z_0_vec = np.zeros(tau_steps+1, dtype="complex")
        Z_0_vec[tau_steps] = Z_0


    # Hacemos una primera invocación para tener precompilada la funcion
    rk4_numba(Z_0_vec, dt, 1, rox, roy, mu, K, tau_steps)

    # Medimos el tiempo de aplicar rk4 durante N_steps iteraciones
    t0 = time.time()
    frames = rk4_numba(Z_0_vec, dt, N_steps, rox, roy, mu, K, tau_steps)
    print(f"{round(time.time()-t0, 5)} segundos para {N_steps} iteraciones.\nEso es {round((time.time()-t0)/N_steps, 10)} s/iteration")
    save_data(frames, "last_tau_vector", data_dir, slice(-10*tau_steps,None,None))
    #  save_data(frames, f"K_{K}_Tau_{tau}_frames", data_dir)#, slice(-40*tau_steps,None,None))

    period_n, iterations_period = period(frames[-150000:], return_iterations_period=True)
    period_time = iterations_period*dt
    print(f"Óribita de periódo {period_n} de duración {1000*period_time} milisegundos")
    print(f"{150000/iterations_period} vueltas totales")



    xframes = list(map(lambda z: z.real, frames))
    yframes = list(map(lambda z: z.imag, frames))

    #  fig, [ax, ax2] = plt.subplots(1,2)
    fig, ax = plt.subplots(1,1)
    ax.axis([-0.05,1.05,-0.05,1.05])
    #  ax2.axis([0,1,0,1])
    points, = ax.plot(xframes[0], yframes[0], 'r-');
    front, = ax.plot(xframes[0], yframes[0], 'bo');

    ax.set_xlabel(r"$x$", fontsize=16)
    ax.set_ylabel(r"$y$", fontsize=16, rotation=0)
    ax.tick_params(axis='both', which='major', labelsize=16)



    def animate(i, xframes, yframes, points, front, animation_jump=10):
        jump = animation_jump
        x = [xx for xx in xframes[:i*jump+1]]
        y = [yy for yy in yframes[:i*jump+1]]
        xlast = x[-1]
        ylast = y[-1]
        points.set_data(x, y)
        front.set_data(xlast, ylast)
        return points, front

    anim = FuncAnimation(fig, animate, frames=N_steps,
                         fargs=(xframes, yframes, points, front, animation_jump),
                         interval=1, blit=True)

    plt.show()
