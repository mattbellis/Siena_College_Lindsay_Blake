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
Nsig = 100
sig_mean = 10.1
sig_width = 0.05
signal = np.random.normal(sig_mean,sig_width,Nsig)

# So here's your background data!
Nbkg = 1000
background = 9.0+(2*np.random.random(Nbkg))

# Combine the background and signal, because when we run the experiment, we actually
# don't know which is which!
data = signal.copy()
data = np.append(data,background.copy())

# Here's a very simple plot of our data.
plt.figure()
plt.hist(data,bins=50)#,range=(60,100))
#plt.show()
#exit()


################################################################################
# Question #1
################################################################################
# Your total probability will now be composed of two separate probabilities:
# * the probability that an event came from signal
# * the probability that an event came from bacground
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
# Use this information to fit 


# YOUR WORK HERE.
print 'Question 1: Calculate the probabilities of obtaining these scores from a gaussian with a width of 10 and mean of 90. \n \n'

my_mean = 90
my_width = 10
probabilities = []
log_probabilities = []

for i in range(Nstudents):
    my_x = scores[i]
    h = Gaussian(my_x, my_mean, my_width)
    probabilities.append(h)
    log_probabilities.append(math.log(h))
    
print (probabilities)    

################################################################################
# Question #2
################################################################################
# What is the product of those probabilities?

# YOUR WORK HERE.
print '\n \nQuestion 2: Taking the product of the probabilities is difficult because all the numbers are so small.  So the way to to take the product is to take the natural log and the sum all of the factors.'

my_sum = sum(log_probabilities)
print '\n \n', my_sum
################################################################################
# Question #3
################################################################################
# Vary the mean and width to find the maximum probability of measuring those
# particular test scores.

# YOUR WORK HERE.
print '\n \nQuestion 3:  Explore the parameter space!  Very the parameters to maximise the sum of the log of the product of probabilities.'

#variables to loop: x, mean, and width
#for each new set create a list of the log of probabilities
#then sum the probabilities, and keep track of which is GREATEST

maxLogProb = -1e6
max_width = 0
max_mean = 0

#mean can range from a score of 0 to 100
for mean in range(101):
    ps_probabilities = []
    ps_mean = mean
#width can range from a value of zero to twenty
    for width in range(2,23):
        ps_width = width
#x can range from 0 to the number of students
        for x in range(Nstudents):
            score = scores[x]
            logd = math.log(Gaussian(score, ps_mean, ps_width))
            ps_probabilities.append(logd)
            sum_probs = sum(ps_probabilities)
            
            if maxLogProb < sum_probs:
                maxLogProb = sum_probs
                max_width = width
                max_mean = mean
    
print '\n \n', maxLogProb
print max_width
print max_mean


x2 = np.linspace(65,95,1000)
y2 = Gaussian(x2,max_mean,max_width)
plt.figure()
plt.hist(scores,bins=25)

plt.figure()
plt.plot(x2,y2)
plt.show()




