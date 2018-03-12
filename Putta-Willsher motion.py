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
    for x in range(Iterations):
        pid = current_process()._identity[0]
        randst = np.random.mtrand.RandomState(pid)
        LEFT = []
        RIGHT = []
        for y in np.linspace(0, radius, collisions):
            MBleft = randst.choice(v, p=MaxwellBoltzmann(v, Temp, m))
            MBright = randst.choice(v, p=MaxwellBoltzmann(v, Temp, m))

            LEFT.append(abs(MBleft * m / M) if abs(MBleft) >= y / dt else 0)
            RIGHT.append(-abs(MBright * m / M) if abs(MBright) >= y / dt else 0)

        out = [LEFT[i] + RIGHT[i] for i in range(len(LEFT))]
        queue.put(np.sum(out))
    queue.close()

"""Helium"""
Vmax = 2500 #m/s
dt = 10**(-9)
radius = Vmax*dt
volume = (4/3)*np.pi*radius**3
particles = volume*scp.N_A/22400
collisions = np.round(particles/6)

m=6.6464764*10**(-27) #Gas molecule mass
M=m*10**22 #Particle mass
Temp = 110  # Kelvin


I=1000#Number of Iterations
Ipt=200 #Number of iterations per thread
v = np.linspace(-4500,4500,9001)

ThreadCount = math.ceil(I / Ipt)
queue = Queue()
threads = [Process(target=Generate, args=(Ipt, queue)) for _ in range(ThreadCount)]

[p.start() for p in threads]
[p.join() for p in threads]

Velocity = []
while not queue.empty():
    Velocity.append(queue.get())
print(len(Velocity))



Brown=np.cumsum(Velocity)

NewV = np.diff(Brown)
GasV = NewV
gauss=np.histogram(Velocity,I)

HistV = gauss[1][:-1]
HistP = gauss[0]


#N=np.nonzero(HistP)[0].size
N=np.sum(HistP)
#print(N)

variance=np.sum(np.power(HistV[np.nonzero(HistP)],2))/(np.pi*I)
#print(variance**0.5)
#print(np.power(HistV[np.nonzero(HistP)],2))


fig = plt.figure(figsize=(15, 7.5))

ax1 = plt.subplot2grid((3,2),(0,0), colspan=2)
ax1.plot(np.array(range(Brown.size)), Brown)

ax2 = plt.subplot2grid((3,2),(1,0), colspan=2)
ax2.plot(np.array(range(Brown.size)), Velocity, color="g")

ax3 = plt.subplot2grid((3,2),(2,0), colspan=2)
ax3.plot(HistV,HistP/I, color="black")
#ax3.plot(v,Gaussian(v,variance,0), color="red", linestyle="--", linewidth=2)
#ax3.plot(v,MaxwellBoltzmann(v,Temp,m))


#np.savetxt("x",gauss[1][:-1])
#np.savetxt("y",gauss[0]/I)
#np.savetxt("position", Brown)

plt.show()