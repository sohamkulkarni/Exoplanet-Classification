import numpy as np
import scipy.stats as stat
import math
import matplotlib.pyplot as plt
from astropy.io import fits
from astroML.plotting import plot_tissot_ellipse
from matplotlib.patches import Ellipse
from sklearn.mixture import GMM


#---Data Files

data1 = np.genfromtxt('planetsyo.csv', usecols = (1,2,4,10,11),dtype=(float,float,float,float, "|S24"), delimiter=",", unpack = 1) #Has all the parameters
data2 = np.genfromtxt('exoplanet.eu_catalog(1).csv',usecols = (0),dtype=("|S24"), delimiter=",", unpack = 1) #Using this to cross reference and take only the planets common to both
# esi = np.genfromtxt('esitotal.txt')


#---Part of the ESI calculation.

def index(x,x0,w):
    return (1-abs((x-x0)/(x+x0)))**w


#---Generating the data

data = list()
data_lat = list()
for i in range (0,len(data1)):
    for j in range (0,len(data2)):
        if data1[i][4]==data2[j]:             #---Comparing the namesof the planets from the two datasets.
#             if (data2[j][3]==2016 or data2[j][3]==2015): data_lat.append(data2[j])
#             if (np.isnan(data2[j][4])==0):
#                 data.append(data2[j])
            data.append(data1[i]) 



mass = np.zeros(len(data))
radius = np.zeros(len(data))
# ecc = np.zeros(len(data))
period = np.zeros(len(data))
temp = np.zeros(len(data))
# mass_lat = np.zeros(len(data_lat))
# radius_lat = np.zeros(len(data_lat))


for i in range(0,len(data)):
    period[i] = data[i][0]
    mass[i] = data[i][1]
    radius[i] = data[i][2]
    temp[i] = data[i][3]
#     ecc[i] = data[i][4]
N=len(data)
# ecce = ecc

# j=0
# count = 0
# for i in range(0,len(data)):
#     slope = (np.log(1.8986e27*data[i][0])-np.log(5e23))/(np.log(69911*data[i][1])-np.log(1.7e3))
#     if(slope <= 3.3):
#         mass[j] = data[i][0]
#         radius[j] = data[i][1]
#         j=j+1
# N=j
    
# print(j)
# for i in range(0,len(data_lat)):
#     mass_lat[i] = data_lat[i][0]
#     radius_lat[i] = data_lat[i][1]
    
# density = np.log(0.75*mass/(1000*np.pi*(radius**3.0)))

density = 1.32717609*mass/(radius**3.0)
vesc = np.sqrt(mass/radius)*(0.601e5/1.1184e4) #-- (All in Earth Units)
gsurf = 2.6406*mass/(radius**2) #--(Earth Units)
rad = radius*10.9733 #--(R_jupiter to R_earth conversion)
per = period/365.25 #--Earth Units
den = density/5.51 #--Earth units


# ecc.resize(N,1)
# c = np.concatenate((density,ecc),axis=1)
# density_lat = 1.32717609*mass_lat/(radius_lat**3.0)
# density_lat.resize((len(density_lat),1))


# In[6]:


#---All ESI calculations.

esir = index(rad,1,0.57)
esid = index(den,1,1.07)
esig = index(gsurf,1,0.13)
esiev = index(vesc,1,0.7)
esit = index(temp,288,5.58)
esip = index(per,1,0.7)
esisurf = (esig*esiev*esit*esip)**0.25
esiint = (esir*esid)**0.5
esitotal = (esisurf*esiint)**0.5
esitotal2 = (esig*esiev*esit*esip*esir*esid)**(1./6)
esi3 = (esiev*esit*esir*esid)**(0.25)
esitotal.resize(N,1)
esitotal2.resize(N,1)   #---This is the one used in the paper.
esi3.resize(N,1)

density.resize(N,1)
density = np.log10(density)

fin = np.concatenate((density,esitotal2),axis=1) #---Aligning in the format required by GMM.fit

#---AIC, BIC calculation using GMM

N = np.arange(1, 5)
models = [None for i in range(len(N))]

for i in range(len(N)):
    models[i] = GMM(N[i]).fit(density)

AIC = [m.aic(density) for m in models]
BIC = [m.bic(density) for m in models]

#-- The above code will give a huge warning about the deprecation of GMM, just a warning.


#--- Plots for AIC/BIC

plt.plot(N, AIC, '-b', label='AIC',linewidth = 1.5)
plt.plot(N, BIC, '--g', label='BIC',linewidth = 1.5)
plt.xlabel('Number of components', fontsize=16)
plt.ylabel('Information criterion', fontsize=16)
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.legend(loc='upper left')
# plt.savefig('aicbic_including_points.png')
# plt.savefig('denecc.png')
# plt.savefig('esi_den2.png')
plt.show()

#---Some required fit parameters

for i in range(3):
    print('Mean:',10**(models[i].means_),'covariance:',(models[i].covars_),'weights:',(models[i].weights_))
    print('Meanreal:',(models[i].means_))


#--- One dimensional density histogram.


plt.figure(1)
# pl.hist(density, bins=np.logspace(-2,4,30))
# pl.gca().set_xscale("log")
# pl.savefig('histogram.png')
x=[-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5]
y=np.zeros(len(x))
for i in range(len(x)):
    y[i] = round(10**(x[i]),2)

n, bins, patches = plt.hist(density, 32, facecolor='g', alpha=0.9,normed = True)
# plt.plot(np.linspace(-2,4,1000),dist3,color='black',linewidth = '1.5')
# plt.plot(np.linspace(-2,4,1000),dist3,color='red')
# plt.plot(np.linspace(-2,2,1000),25*(stat.norm.pdf(np.linspace(-2,2,1000),loc=-0.05755607,scale=0.20218831)),'b')
# plt.plot(np.linspace(-2,4,1000),25*(stat.norm.pdf(np.linspace(-2,4,1000),loc=0.98637232,scale=1.08479183)),'b')
plt.xticks(x, y, fontsize='10')
plt.xlabel('Density $(gm/cm^3)$',fontsize=16)
plt.ylabel('Normalised Frequency',fontsize=16)
# plt.savefig('histogram_including_points.png')
# plt.savefig('3ghist.png')

plt.show()

#---2D Histogram plot

# den_bin = [-1,-0.5,0,0.5,1,1.5,2]
# ecc_bin = [0,0.2,0.4,0.6,0.8,1]
den_bin = 40
# ecc_bin = 20
esi_bin = 40
dd = 1.32717609*mass/(radius**3.0)
# den = np.log10(1.32717609*mass/(radius**3.0))
# ecce = ecc.resize(1,N)
# esitt = (esisurf*esiint)**0.5
esitt2 = (esig*esiev*esit*esip*esir*esid)**(1./6)
H, den_bin, esi_bin = np.histogram2d(np.log10(dd),esitt2,bins=(den_bin, esi_bin))


#---Real density plot

# plt.loglog(radius*69911, mass*1.8986e27,'.r')
# plt.loglog(radius_lat*69911, mass_lat*1.8986e27,'.b',label='2015&2016')
# plt.legend(loc='lower right')
plt.plot(np.log10(dd),esitt2,'.')
# plt.plot((1.7e3,1e5),(5e23,1e29),'g')
# plt.plot((4.5e3,5e5),(1e23,8e28),'g')
# plt.xlim((1.7e3,5e5))
# plt.ylim((1e23,1e29))
plt.xlabel('Radius(Km)',fontsize=16)
plt.ylabel('Mass(Kg)',fontsize=16)
# plt.savefig('fig1.png')
plt.show()


#---Elliptical plots

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
# alpha = 1/510
v = [0.135,0.00114]
v2 = [0.498,0.001234]
v3 = [0.1941, 0.00128]

def alp(var):
    ang = np.arctan(var[1] / var[0])
    ang = 180. * ang / np.pi
    return ang

# w=v
# ang = np.arctan(w[1] / w[0])
# ang = 180. * ang / np.pi
v = np.sqrt(v)
v2 = np.sqrt(v2)
v3 = np.sqrt(v3)
mean = np.array([-0.2167,0.0424])
mean2 = np.array([2.27,0.0396])
mean3 = np.array([0.57,0.06])
ax.plot(np.log10(dd),esitt2,'.',color='black', ms=6)
# ax.imshow(H.T, interpolation='nearest', origin='lower', extent=[den_bin[0], den_bin[-1], esi_bin[0], esi_bin[-1]],cmap = plt.cm.binary,aspect='auto')
# plt.ylim(0,1)
colors=['cyan','yellow']
# for i in (1,2):
ax.add_patch(Ellipse(mean, 2*v[0], 2*v[1],angle=180+alp(v), lw=1,color='cyan'))
ax.add_patch(Ellipse(mean2, 2*v2[0], 2*v2[1],angle=180+alp(v2), lw=1,color='yellow'))
ax.add_patch(Ellipse(mean3, 2*v3[0], 2*v3[1],angle=180+alp(v3), lw=1,color='red'))

print(alp(v),alp(v2),alp(v3))

kwargs = dict(ha='left', va='top', transform=ax.transAxes)
plt.xlabel('log(density[$gm/cm^3$]) ',fontsize=20)
plt.ylabel('Total ESI',fontsize=20)
# plt.savefig('eccvsden.png')
# ,angle=alpha * 180. / np.pi
# plt.savefig('denvsecc2.png')
plt.savefig('temp.png')
# plt.savefig('denesi2.png')
plt.show()


#--- Elliptical plots end here


#--- Trying to fit the gaussian on the density histogram.(Not used anywhere.)

distr = (0.283*stat.norm.pdf(np.linspace(-2,4,1000),loc=0.98637232,scale=np.sqrt(1.08479183)))+(0.716*stat.norm.pdf(np.linspace(-2,4,1000),loc=-0.05755607,scale=np.sqrt(0.20218831)))
dist3 = (0.508*stat.norm.pdf(np.linspace(-2,4,1000),loc=-0.14657575,scale=np.sqrt(0.16568)))+(0.3929*stat.norm.pdf(np.linspace(-2,4,1000),loc=0.3069505,scale=np.sqrt(0.36470)))+(0.0988696*stat.norm.pdf(np.linspace(-2,4,1000),loc=1.94438377,scale=np.sqrt(0.82264)))
dist31 = 0.508*stat.norm.pdf(np.linspace(-2,4,1000),loc=-0.14657575,scale=np.sqrt(0.16568))

#--- Individual histograms

fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(1,2,1)
ax.hist(esitt2,bins=45,color='green')
ax.set_xlabel('esi_total', fontsize=20)

ax = fig.add_subplot(1,2,2)
ax.hist(np.log10(dd),bins=45,color='green')
ax.set_xlabel('log(density$[gm/cm^3]$)', fontsize=20)
plt.savefig('hist_individ_2.png')
plt.show()
