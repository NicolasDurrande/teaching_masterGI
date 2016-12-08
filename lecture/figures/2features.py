import numpy as np
import pylab as pb
pb.ion()

####################################################
# example
x = np.arange(0,20)
y = x/2. + np.cos(x) + np.random.normal(0,2,20)

pb.figure(figsize=(10,4))
pb.plot(x,y,"bx",mew=1.5)
pb.ylim((-10,20))
pb.savefig("2_example.pdf",bbox_inches="tight")

####################################################
# centered
x = np.arange(0,20)
y = np.cos(x) + np.random.normal(0,2,20)

pb.figure(figsize=(5,4))
pb.plot(x,y+10,"bx",mew=1.5)
pb.ylim((-20,20))
pb.savefig("2_centredno.pdf",bbox_inches="tight")

pb.figure(figsize=(5,4))
pb.plot(x,y,"bx",mew=1.5)
pb.ylim((-20,20))
pb.savefig("2_centred.pdf",bbox_inches="tight")


####################################################
# trend
x = np.arange(0,20)
y = np.cos(x) + np.random.normal(0,2,20)

pb.figure(figsize=(5,4))
pb.plot(x,y,"bx",mew=1.5)
pb.ylim((-10,25))
pb.savefig("2_trendno.pdf",bbox_inches="tight")

pb.figure(figsize=(5,4))
pb.plot(x,y+x,"bx",mew=1.5)
pb.ylim((-10,25))
pb.savefig("2_trend.pdf",bbox_inches="tight")

####################################################
# seasonal
x = np.arange(0,20)
y = np.random.normal(10,2,20)

pb.figure(figsize=(5,4))
pb.plot(x,y,"bx",mew=1.5)
pb.ylim((0,20))
pb.savefig("2_seasonalno.pdf",bbox_inches="tight")

pb.figure(figsize=(5,4))
pb.plot(x,0.5*y + 4*np.cos(x) +5,"bx",mew=1.5)
pb.ylim((0,20))
pb.savefig("2_seasonal.pdf",bbox_inches="tight")

####################################################
# cyclical
x = np.arange(0,20)
y = np.random.normal(10,2,20)
z = 0*x
z[:3] = (-13,-10.5,-4.5)
pb.figure(figsize=(5,4))
pb.plot(x,y,"bx",mew=1.5)
pb.ylim((0,20))
pb.savefig("2_cyclicalno.pdf",bbox_inches="tight")

pb.figure(figsize=(5,4))
pb.plot(x,0.5*y + 4*np.cos(x) +5+z,"bx",mew=1.5)
pb.ylim((0,20))
pb.savefig("2_cyclical.pdf",bbox_inches="tight")

####################################################
# stationnary
x = np.arange(0,20)
y = np.random.normal(0,1,20)

pb.figure(figsize=(5,4))
pb.plot(x,x*y/2.,"bx",mew=1.5)
pb.ylim((-20,20))
pb.savefig("2_stationnaryno.pdf",bbox_inches="tight")

pb.figure(figsize=(5,4))
pb.plot(x,5*y,"bx",mew=1.5)
pb.ylim((-20,20))
pb.savefig("2_stationnary.pdf",bbox_inches="tight")

####################################################
# Noise
x = np.arange(0,20)
y = x/2. + 4* np.cos(x/4.)

pb.figure(figsize=(5,4))
pb.plot(x,y,"bx",mew=1.5)
pb.ylim((-0,15))
pb.savefig("2_noisyno.pdf",bbox_inches="tight")

pb.figure(figsize=(5,4))
pb.plot(x,y+np.random.normal(0,1,20),"bx",mew=1.5)
pb.ylim((-0,15))
pb.savefig("2_noisy.pdf",bbox_inches="tight")

####################################################
# missing
x = np.arange(0,20)
y = 0.5*np.random.normal(10,2,20) + 4*np.cos(x) +5

# pb.figure(figsize=(5,4))
# pb.plot(x,y,"bx",mew=1.5)
# pb.ylim((0,20))
# pb.savefig("2_missingno.pdf",bbox_inches="tight")

z = 0*x
z[3] = 50
z[8] = 50
z[9] = 50
z[10] = 50
z[11] = 50
pb.figure(figsize=(9,4))
pb.plot(x,y+z,"bx",mew=1.5)
pb.ylim((0,20))
pb.savefig("2_missing.pdf",bbox_inches="tight")

####################################################
# outliers
x = np.arange(0,20)
y = 0.5*np.random.normal(10,2,20) + 4*np.cos(x) +5
z = 0*x
z[9] = 112
pb.figure(figsize=(9,4))
pb.plot(x,y+z,"bx",mew=1.5)
pb.ylim((-10,130))
pb.savefig("2_outliers.pdf",bbox_inches="tight")

