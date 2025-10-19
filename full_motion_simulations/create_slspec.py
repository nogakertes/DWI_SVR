import json
import numpy as np

if __name__ == "__main__":
    MB_factor = 5
    N_slices = 80

    assert N_slices%MB_factor == 0, 'Multiband factor and number of slices are not compatible!' 
    data = np.arange(N_slices).reshape(N_slices//MB_factor, MB_factor)
    # INSERT_YOUR_CODE
    np.savetxt(f'slspec_MB{MB_factor}_S{N_slices}.txt', data, fmt='%d', delimiter=' ')
    print('f')
    
