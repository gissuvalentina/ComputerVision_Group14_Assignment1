#####TUWIEN - CV: Task1 - Scale-Invariant Blob Detection
#####*********+++++++++*******++++INSERT GROUP NO. HERE
from typing import Tuple
import numpy as np
import cv2


def create_log_kernel(size: int, sig: float) -> float:
    """
    Returns a rotationally symmetric Laplacian of Gaussian kernel
    with given 'size' and standard deviation 'sig'

    Parameters
    ----------
    size : int
        size of kernel (must be odd) (int)
    sig : int
        standard deviation (float)
    
    Returns
    --------
    float
        kernel: filter kernel (size x size) (float)

    """

    kernel = np.zeros((size, size), np.float64)
    halfsize = int(np.floor(size / 2))
    r = range(-halfsize, halfsize + 1, 1)
    for x in r:
        for y in r:
            hg = (np.power(np.float64(x), 2) + np.power(np.float64(y), 2)) / (2 * np.power(np.float64(sig), 2))
            kernel[x + halfsize, y + halfsize] = -((1.0 - hg) * np.exp(-hg)) / (np.pi * np.power(sig, 4))

    return kernel - np.mean(kernel)


def get_log_pyramid(img: np.ndarray, sigma: float, k: float, levels: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a LoG scale space of given image 'img' with depth 'levels'
    The filter parameter 'sigma' increases by factor 'k' per level
    HINT: np.multiply(..), cv2.filter2D(..)

    Parameters
    ----------
    img : np.ndarray
        input image (n x m x 1) (float)
    sigma : float
        initial standard deviation for filter kernel
    levels : int
        number of layers of pyramid 

    Returns
    ---------
    np.ndarray
        scale_space : image pyramid (n x m x levels - float)
    np.ndarray
        all_sigmas : standard deviation used for every level (levels x 1 - float)
    """

    # student_code start
    
    if img.ndim == 3 and img.shape[2] == 1:
        img_2d = img[:, :, 0]
    else:
        img_2d = img

    h, w = img_2d.shape[:2]

    # (n x m x levels)
    scale_space = np.zeros((h, w, levels), dtype=np.float32)
    # sigmas used at each level
    all_sigmas = np.zeros(levels, dtype=np.float32)

    cur_sigma = sigma

    for i in range(levels):
        # filter size: 2 * floor(3*sigma) + 1   (covers [-3σ, 3σ])
        size = int(2 * np.floor(3 * cur_sigma) + 1)

        # 1) LoG kernel at current scale
        log_kernel = create_log_kernel(size, cur_sigma)

        # scale-normalize: multiply by σ²
        log_kernel = np.multiply(log_kernel, cur_sigma ** 2)

        # 2) convolve and take absolute response
        response = cv2.filter2D(
            img_2d.astype(np.float32),
            ddepth=-1,
            kernel=log_kernel,
            borderType=cv2.BORDER_REPLICATE,
        )

        scale_space[:, :, i] = np.abs(response)
        all_sigmas[i] = cur_sigma

        # 3) increase σ by factor k
        cur_sigma *= k

    # student_code end

    return scale_space, all_sigmas
