from numba import cuda
import numba
import numpy as np
import math
import matplotlib.pylab as plt


#npts = 10000
#my_gpu = numba.cuda.get_current_device()

#np.random.seed(1)

#x = 10*np.random.random(npts)
#y = 10*np.random.random(npts)
#x = np.float32(x)
#y = np.float32(y)
#ans = np.zeros(npts, dtype = np.float32)

@numba.cuda.jit("void(float32[:],float32[:],float32[:],float32[:],float32[:],float32)")
def compute(arr_ax, arr_ay, arr_bx, arr_by, arr_out, k):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw
    if i>= arr_out.size:
        return
    a0x = arr_ax[i]
    a0y = arr_ay[i]
    #sort(a0x)
    #arr_out[0] = arr_ax[0]
    #return
    narr_b = len(arr_bx)
    #distance = np.array(narr_b)
    for j in xrange(narr_b):
        diffx = a0x - arr_bx[j]
        diffy = a0y - arr_by[j]
        #arr_out[i] = i
        #arr_out[i] = a0y
        #arr_out[0] = arr_bx[0]
        #'''
        distance = math.sqrt(diffx * diffx + diffy * diffy)
        #distances.sort()
        if distance<=k:
            #arr_out[i] = distance
            arr_out[i] += 1
        #'''
    arr_out[i] /= (k*narr_b)






#radius = 0.1 #change the accepted radius here
#thread_ct = my_gpu.WARP_SIZE
#block_ct = int(math.ceil(float(npts) / thread_ct))
#compute[block_ct, thread_ct](x, y, ans, radius)

#ans -= 1 #Since each element counts itself
#print ans



