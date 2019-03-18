import numpy as np

def convolute_x(x, f_spatial_ext, stride):
    """
    Convolutes a 2D layer into a x_col layer.

    Parameters
    ----------

    x : 2D Numpy array. Shape (H, W)
        Data to be convoluted.

    f_spatial_ext : int
        Filtersize

    stride : int
        Stride

    Returns
    -------

    x_col : 2D Numpy array
        Convoluted array with each convolution as a column.
    """
    width , height = x.shape
    convolution = list()
    for h_ix in range(0, height-f_spatial_ext+1, stride):
        for w_ix in range(0, width-f_spatial_ext+1, stride):
            convolution.append(x[h_ix:h_ix+f_spatial_ext, w_ix:w_ix+f_spatial_ext].flatten())
    x_col = np.stack(convolution, axis=0).T
    return x_col