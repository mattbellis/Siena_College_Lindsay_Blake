import numpy as np
import matplotlib.pylab as plt

#infilename = "output_of_tests.dat"
#infilename = "output_1k.dat"
#infilename = "output_1k_both_pois.dat"
#infilename = "output_1k_both_pois_minos.dat"
#infilename = "output_1k_nn_0.dat"
#infilename = "output_1k_nn_0_r01.dat"
#infilename = "output_nn_0_r05.dat"
#infilename = "output_mlm_0.dat"
#infilename = "output_nn_0_r01.dat"
infilename = "output_2Dnn_0_r05.dat"
#infilename = "false_output_2Dnn_0_r025.dat"
infile = open(infilename)

fracs =[]
errors = []
true_fracs =[]
for line in infile:
    if line.find('core')<0:
        vals = line.split()
        fracs.append(float(vals[2]))
        errors.append(float(vals[3]))
        true_fracs.append(float(vals[6]))

print errors[0:10]
print fracs[0:10]
print true_fracs[0:10]

errors = np.array(errors)
fracs = np.array(fracs)
true_fracs = np.array(true_fracs)

def Gaussian(x,mean,width):

    y = (1.0/(width*np.sqrt(2*np.pi)))*np.exp(-((x-mean)**2)/(2*(width**2)))

    return y




print "Fracs:"
print np.mean(fracs)
print np.std(fracs)

print "Errors:"
print np.mean(errors)
print np.std(errors)

#pulls = abs(fracs-true_fracs)/(errors/1.4)
#pulls = abs(fracs-0.9)/(errors/1.0)
pulls = (fracs-0.90)/(errors/1.0)

print "Pulls:"
print np.mean(pulls)
print np.std(pulls)

#fraction plot
plt.figure(figsize=(6,9),dpi=100)
plt.subplot(3,1,1)
plt.title('Fraction Plot')

nbins = 50
Nfrac = len(fracs)
lo = 0.85
hi = 0.95
xF = np.linspace(lo,hi,Nfrac)
binwidth = (hi-lo)/float(nbins)

frac_mean = .9
frac_width = .005
#frac_width = 0.013
fracGauss = Gaussian(xF,frac_mean,frac_width)
fracGauss *= len(fracs)*binwidth

plt.hist(fracs,bins=nbins,range=(lo,hi))
plt.plot(xF,fracGauss)

#pull plot
plt.subplot(3,1,2)
plt.title('Pull Plot')

pullhi = 4
pulllo = -4
Npull = len(pulls)
xP = np.linspace(pulllo,pullhi,Npull)
pullbinwidth = (pullhi-pulllo)/float(nbins)

pullGauss = Gaussian(xP,0,1)
pullGauss *= len(pulls)*pullbinwidth

plt.plot(xP,pullGauss)
plt.hist(pulls,bins=nbins,range=(pulllo,pullhi))

#error subplot
plt.subplot(3,1,3)
plt.title('Error Plot')

err_mean = np.mean(errors)
err_width = np.std(errors)

errhi = max(errors)
errlo = min(errors)
Nerr = len(errors)
errbinwidth = (errhi-errlo)/float(nbins)
xE = np.linspace(errlo,errhi,Nerr)

errGauss = Gaussian(xE,err_mean,err_width)
errGauss *= len(errors)*errbinwidth

plt.hist(errors,bins=nbins)
plt.plot(xE,errGauss)

plt.show()
