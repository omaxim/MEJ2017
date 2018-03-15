import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

def sortAlg(a, b):
    return a[0] - b[0]
def lnfit(array,A,B):
    return A*np.log(B*array)

k = 1.38064852 * 10 ** (-23)

gasses = [
(40,"A"),
(36,"K"),
(48,"H"),
(46,"N"),
(32,"X"),
]



for (count, prefix) in gasses:
    MBVAR = []
    MODVAR = []
    T = []
    for n in range(count):
        with open(prefix+str(n)+".txt","r") as data:
            header = data.readline()
            print(header)
            preambule={}
            for pair in header[1:-1].split(sep=","):
                #print(pair.split("="))
                [key,value]=pair.split("=")
                preambule[key]=float(value)
            xstrings=data.readline()[1:-1]
            positions=np.array([float(i) for i in xstrings.split(", ")])
        m = preambule["gass_mass"]
        temp = preambule["temp"]
        I = preambule["iterations"]
        boltzmann_var = (k*temp)/m

        bincount=1000
        velocity=np.diff(positions)
        gauss=np.histogram(velocity,bincount)   #DO NOT NORM
        HistV = gauss[1][:-1]
        HistP = gauss[0]


        putta_variance = np.sum(np.multiply(np.power(HistV[np.nonzero(HistP)],2), HistP[np.nonzero(HistP)]))/I
        MBVAR.append(boltzmann_var)
        MODVAR.append(putta_variance)
        T.append(temp)



    linked = sorted(list(zip(MBVAR, MODVAR, T)), key=lambda a: a[1])

    np.save(prefix,linked)

"""Fitting"""
#x = np.linspace(0,1.4*10**(-17),1000)
#xdata = [i[0] for i in linked]
#ydata = [i[1] for i in linked] + 10**(-18) * np.random.normal(size=len(xdata))
#guess = [1/0.0000683672,1/(4.15*10**(-19))]
#fit,_= spo.curve_fit(lnfit,xdata,ydata,guess)

#print(fit[0])
#print(fit[1])
#plt.plot([i[1] for i in linked], [i[0] for i in linked], marker=".")
#plt.plot(x,lnfit(x,guess[0],guess[1]))
#plt.plot(x,lnfit(x,fit[0],fit[1]), color="red")
#plt.axis((0,1.4*10**(-17),10000,55000))
#plt.scatter([i[1] for i in linked], [i[0] for i in linked])
#plt.show()