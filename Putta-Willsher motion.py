import numpy as np
import random as rn
import scipy.constants as scp
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, current_process
import math

def MaxwellBoltzmann(Array, Temperature, Mass):
    k = 1.38064852 * 10 ** (-23)  # Boltzmann constant
    v = Array
    return ( ( Mass / ( 2 * np.pi * k * Temperature ) ) ** ( 1 / 2 )) * np.exp( - ( Mass * v ** 2 ) / ( 2 * k * Temp ) )

def Gaussian(Array,Variance, Center):
    return (1/np.sqrt(2*np.pi*Variance))*np.exp(-(Array-Center)**2/(2*Variance))

def Generate(Iterations, queue):
    Velocity = []
    pid = current_process().pid
    rs = np.random.RandomState(pid)
    for x in range(Iterations):
        LEFT = []
        RIGHT = []
        for y in np.linspace(0, radius, collisions):
            MBleft = rs.choice(v, p=MaxwellBoltzmann(v, Temp, m))
            MBright = rs.choice(v, p=MaxwellBoltzmann(v, Temp, m))

            LEFT.append(abs(MBleft * m / M) if abs(MBleft) >= y / dt else 0)
            RIGHT.append(-abs(MBright * m / M) if abs(MBright) >= y / dt else 0)

        out = [LEFT[i] + RIGHT[i] for i in range(len(LEFT))]
        Velocity.append(np.sum(out))
    queue.put(Velocity)
    queue.close()

"""Helium"""
Vmax = 2500 #m/s
dt = 10**(-9)
radius = Vmax*dt
volume = (4/3)*np.pi*radius**3
particles = volume*scp.N_A/22400
collisions = np.round(particles/6)

m=6.6464764*10**(-27) #Gas molecule mass
M=10**(-11) #Particle mass
Temp = 300  # Kelvin
print((M/m)**2)

I=2000#Number of Iterations
Ipt=400 #Number of iterations per thread
v = np.linspace(-4500,4500,9001)

ThreadCount = math.ceil(I / Ipt)
queue = Queue()
threads = [Process(target=Generate, args=(Ipt, queue)) for _ in range(ThreadCount)]

[p.start() for p in threads]
[p.join() for p in threads]

Velocity = []
while not queue.empty():
    Velocity += queue.get()

np.save("Velocity",Velocity)
