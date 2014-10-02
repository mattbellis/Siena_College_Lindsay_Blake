import gen_function_density as genRD
#import cdist_calc_distances_Garrett as genRD
import closestNeighborGPU as genRD_GPU
import gen_data_and_templates as genD
import scipy.stats as stats
import numpy as np
from numba import cuda
import numba
import math
from time import time

np.random.seed(1)

Nsig = 100

Nsig = stats.poisson(Nsig).rvs(1)[0]
sig_mean1 = 5.0
sig_width1 = 0.5
sig_mean2 = 15.0
sig_width2 = 1.0

Nbkg = 900
Nbkg = stats.poisson(Nbkg).rvs(1)[0]
bkglo1 = 0
bkghi1 = 10
bkglo2 = 10
bkghi2 = 20

Ntemplates = 1000000

true_frac = Nbkg/float(Nsig+Nbkg)

# actually generating data and templates
data, sig_template, bkg_template = genD.sig_ngauss_bkg_flat([sig_mean1, sig_mean2],[sig_width1, sig_width2],Nsig,[bkglo1, bkglo2],[bkghi1,bkghi2],Nbkg,Ntemplates,verbose=False,scale=True)

k=20

my_gpu = numba.cuda.get_current_device()

npts = len(data[0])
distances = np.zeros(len(sig_template[0]), dtype = np.float32)
ans = np.zeros(npts, dtype = np.float32)

#print ans
#print npts

#print "here"
#print len(data[0])
#print data[0][0:10]
#print data[1][0:10]
#print len(sig_template[0])
#print sig_template[0]

radius = 0.01 #change the accepted radius here
thread_ct = my_gpu.WARP_SIZE
block_ct = int(math.ceil(float(npts) / thread_ct))
#data[0] = np.float32(data[0])
#data[1] = np.float32(data[1])
#sig_template[0] = np.float32(sig_template[0])
#sig_template[1] = np.float32(sig_template[1])

start = time()
genRD_GPU.compute[block_ct, thread_ct](np.float32(data[0]), np.float32(data[1]), np.float32(sig_template[0]), np.float32(sig_template[1]), ans, radius)
print "GPU:  %f" % (time()-start)

print ans[0:10]
print len(ans)


start = time()
presigFD = genRD.numInRange(data, sig_template,radius,len(data[0]))
#presigFD = genRD.numInRange(data, sig_template,radius)
print "cdist:  %f" % (time()-start)
print presigFD[0:10]
print len(presigFD[0:10])




#presigFD = genRD.radiusToK(data, sig_template,k)
#prebkgFD = genRD.radiusToK(data, bkg_template,k)


