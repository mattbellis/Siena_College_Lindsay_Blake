@ i = 0

set nsig = $1
set nbkg = $2

while ( $i < 1000 )

    #python mlm_to_compare_with_nn_0.py |& grep FIT
    #python nearest_neighbors_fitting_0.py |& grep FIT
    #python 2D_NN_wModule_finalvalues.py |& grep FIT
    #python 2D_NN_wModule_finalvalues.py   
    #python 2D_mlm.py |& grep FIT
    python 2D_NN_varyradius.py $nsig $nbkg |& grep FIT

    @ i += 1

end
