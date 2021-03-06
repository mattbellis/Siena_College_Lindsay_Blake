import numpy as np
import matplotlib.pylab as plt
import math as math

################################################################################
# This is how we define our own function
################################################################################
def Gaussian(x,mean,width):

    y = (1.0/(width*np.sqrt(2*np.pi)))*np.exp(-(x-mean)**2/(2*(width**2)))

    return y


################################################################################
# First thing I'm going to do is to generate some fake data for us to work with.
# 
# This is going to be more like a particle physics experiment. Suppose I'm looking
# at the mass calculated by combining two particles. Sometimes those two particles
# came from some random process (background), but sometimes they came from 
# some new particle we are hunting for (signal)!
#
# Let's generate these data!
################################################################################

# So here's your signal data!
Nsig = 50
sig_mean = 10.1
sig_width = 0.05
signal = np.random.normal(sig_mean,sig_width,Nsig)

# So here's your background data!
Nbkg = 950
background = 9.0+(2*np.random.random(Nbkg))

# Combine the background and signal, because when we run the experiment, we actually
# don't know which is which!
data = signal.copy()
data = np.append(data,background.copy())

# Here's a very simple plot of our data.
plt.figure()
plt.hist(data,bins=50)


################################################################################
# Question #1
################################################################################
# Your total probability *for each event* will now be composed of two separate 
# probabilities:
#
# 1) the probability that an event came from signal
# 2) the probability that an event came from bacground
# 
# The first part you know how to do (Gaussian). The second part is a bit more challenging, 
# but because the background looks ``flat", it means that every value is equally likely
# over that range (from 9.0 to 11.0). So the probability for an event to come from background
# will be 
#
#    P(bkg) = 1/(range) = 1/(11-9) = 1/2
#
# So even though it's a bit weird, the probability for an event to come from background is
# just 0.5, regardless of where it comes from (so long as it is between 9.0 and 11.0.
#
# You also need to account for the fraction (frac) of the numbers that come from signal or background.
# So your total probability for each event will be....
#
#   P = frac*P(sig)  +  (1-frac)*P(bkg)
#
# Your goal is to vary the mean and width of the Gaussian *and* the fractional amount (frac)
# of signal and background, as described above, to find the values that give you the 
# maximum likelihood. 
#
# Good luck!


# YOUR WORK HERE.

print "I don't know whether the data point in my set is the signal or the \nbackground data. I do know the probabilitiy that it came from either the \nbackground or from a gaussian of a certain mean and width and with a certain \nration of signal to background points.  So what this code does, is varies the fractions, \nand the characteristice of the signal gaussian.  Then for each data point sums \nthe probability it came from the gaussian or from the background, then adds \nthis value to a list.  Then for a whole set of values with specific paramters, \nthe log is taken and each value is summed.  This total sum is then maximized to \nobtain the most likely paramters for the data set that was given to me."


prob_bkg = .5
maxTotProb = -1e6

for mysig_mean in np.arange(9.6,10.5,.1):
    #print mysig_mean
    for mysig_width in np.arange(.01,.1,0.01):
        #print mysig_width
            
        
        
        for frac_bkg in np.arange(0,1,.01):
            
            probabilities = []
            
            frac_sig = 1 - frac_bkg
            #print frac_bkg
            #print frac_sig            
            
            for d in data:
            
                prob_sig = Gaussian(d,mysig_mean,mysig_width)
                prob_bkg = .5
                totPointProb= frac_bkg*prob_bkg+frac_sig*prob_sig
                
                logd = 0.0
            
                if totPointProb>0:
                    logd = math.log(totPointProb)
                    probabilities.append(logd)
                
            totProb = sum(probabilities)
            if maxTotProb < totProb:
                 maxTotProb = totProb
                 maxSigWidth = mysig_width
                 maxSigMean = mysig_mean
                 fracbkg= frac_bkg
                 fracsig = frac_sig   
            
        
            
        
    
    
print maxTotProb
print maxSigWidth
print maxSigMean
print fracbkg
print fracsig

plt.show()

