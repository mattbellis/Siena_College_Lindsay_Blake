from numba import cuda
import numba
import numpy as np
import math
#import matplotlib.pylab as plt


#npts = 10000
#np.random.seed(1)
#my_gpu = numba.cuda.get_current_device()

#ncalc = (npts*(npts-1))/2


#x = 10*np.random.random(npts)
#y = 10*np.random.random(npts)
#x = np.float32(x)
#y = np.float32(y)
#
#z = np.ones(npts*npts, dtype = np.float32) * -1
#final = np.ones(ncalc, dtype = np.float32) * -1
#
my_gpu = numba.cuda.get_current_device()
@numba.cuda.jit("void(float32[:],float32[:],float32[:])")
def compute(arr_a,arr_b,arr_out):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw
    if i>= arr_out.size:
        return
    for j in range(i+1, npts): 
        x0 = arr_a[i]
        y0 = arr_b[i]
        x1 = arr_a[j]
        y1 = arr_b[j]
        xdiff = x0-x1
        ydiff = y0-y1
        arr_out[npts * i + j] = math.sqrt((xdiff * xdiff) + (ydiff * ydiff))
        #arr_out[npts * i + j] = math.sqrt((arr_a[i] - arr_a[j]) * (arr_a[i] - arr_a[j]) + (arr_b[i] - arr_b[j]) * (arr_b[i] - arr_b[j]))



		
thread_ct = my_gpu.WARP_SIZE
block_ct = int(math.ceil(float(ncalc) / thread_ct))
compute[block_ct, thread_ct](x, y, z)
count = 0




#for i in range(0, npts*npts):
#	if z[i] >= 0:
#		final[count] = z[i]
#		count = count + 1



#####print final
z = z[z>-1]  #slicing
print len(z)
print z[0:10]
#x = plt.hist(z, bins=50)

#a = x[0]
#a[0] = 0
#x = x
#print x
#plt.show(x)
