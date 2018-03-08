import numpy as np
import random as rn
import scipy.stats as scp
import matplotlib.pyplot as plt





I=500#Number of Iterations
#Velocity=np.zeros(I)
T=2 #Time
dt=T/I
C=(10**16)/6 #prasarna
m=6.6464764*10**(-27) #Gas molecule mass
M=m*10**22 #Particle mass

Temp = 110  # Kelvin

def MaxwellBoltzmann(Array, Temperature, Mass):
    k = 1.38064852 * 10 ** (-23)  # Boltzmann constant
    v = Array
    return ( ( Mass / ( 2 * np.pi * k * Temperature ) ) ** ( 1 / 2 )) * np.exp( - ( Mass * v ** 2 ) / ( 2 * k * Temp ) )

v = np.linspace(-4500,4500,9001)

def Gaussian(Array,Variance, Center):
    return (1/np.sqrt(2*np.pi*Variance))*np.exp(-(Array-Center)**2/(2*Variance))

Pres=[]


Vmin = 500

p=[]  #Momentum of the large particle
for x in range(I):

    LEFT = []
    RIGHT = []
    for y in range(10):

        MBleft = np.random.choice(v, p=MaxwellBoltzmann(v, Temp, m))
        if np.abs(MBleft) >= 100:
            LEFT.append(MBleft*C*m)
        y = +1

    for z in range(10):
        MBright = np.random.choice(v, p=MaxwellBoltzmann(v, Temp, m))
        if np.abs(MBright) >= 100:
            RIGHT.append(MBright*C*m)

        z=+1

    p.append(np.sum(np.sum(RIGHT),np.sum(LEFT)))
    x=+1


Velocity = np.array(p)/M
Brown=np.cumsum(Velocity)

NewV = np.diff(Brown)
GasV = NewV
gauss=np.histogram((np.array(Velocity)*(M/m)/C),I)

HistV = gauss[1][:-1]
HistP = gauss[0]/I

k=[]
for g in range (HistV.size):
    k.append(((HistV[g]/np.pi)**2)*(HistP[g]))
    g=+1
variance=np.sum(k)
print(variance**0.5)



fig = plt.figure(figsize=(15, 7.5))

ax1 = plt.subplot2grid((3,2),(0,0), colspan=2)
ax1.plot(np.array(range(Brown.size)), Brown)

ax2 = plt.subplot2grid((3,2),(1,0), colspan=2)
ax2.plot(np.array(range(Brown.size)), Velocity, color="g")

ax3 = plt.subplot2grid((3,2),(2,0), colspan=2)
#ax3.plot(HistV,HistP, color="black")
ax3.plot(v,Gaussian(v,variance,0), color="red", linestyle="--", linewidth=2)
ax3.plot(v,MaxwellBoltzmann(v,Temp,m))


#np.savetxt("x",gauss[1][:-1])
#np.savetxt("y",gauss[0]/I)
#np.savetxt("position", Brown)

plt.show()