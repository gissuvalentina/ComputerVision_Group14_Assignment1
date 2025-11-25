#####TUWIEN - CV: Task2 - Image Stitching
#####*********+++++++++*******++++INSERT GROUP NO. HERE
from typing import List
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_simple(images: List[np.ndarray], width: int, height: int, H: List[np.ndarray], T: np.ndarray) -> np.ndarray:
    """
    Stitch the final panorama with the calculated panorama extents
    by transforming every image to the same coordinate system as the center image. Use the dot product
    of the translation matrix 'T' and the homography per image 'H' as transformation matrix.
    HINT: cv2.warpPerspective(..), cv2.addWeighted(..)
    
    Parameters
    ----------
    images : List[np.ndarray]
        list of images
    width : int
        width of panorama (in pixel)
    height : int
        height of panorama (in pixel)
    H : List[np.ndarray]
        list of homographies to center image ([number_of_images x 3 x 3])
    T : np.ndarray
        translation matrix for panorama ([3 x 3])

    Returns
    ---------
    np.ndarray
        (result) panorama image ([height x width x 3])
    """
    
    # student_code start
    result = np.zeros((height, width, 3), dtype=np.uint8)

    for img, H_i in zip(images, H):
        H_total = T @ H_i

        # Warp the image using the combined homography
        warped_img = cv2.warpPerspective(img, H_total, (width, height))

        # Create a mask of non-zero pixels in the warped image
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        mask = gray > 0

        blended = cv2.addWeighted(result, 0.5, warped_img, 0.5, 0)

        # Overlay the warped image onto the result
        result[mask] = blended[mask]
    # student_code end
        
    return result


def get_blended(images: List[np.ndarray], width: int, height: int, H: List[np.ndarray], T: np.ndarray) -> np.ndarray:
    """
    Use the equation from the assignment description to overlay transformed
    images by blending the overlapping colors with the respective alpha values
    HINT: ndimage.distance_transform_edt(..)
    
    Parameters
    ----------
    images : List[np.ndarray]
        list of images
    width : int
        width of panorama (in pixel)
    height : int
        height of panorama (in pixel)
    H : List[np.ndarray]
        list of homographies to center image ([number_of_images x 3 x 3])
    T : np.ndarray
        translation matrix for panorama ([3 x 3])

    Returns
    ---------
    np.ndarray
        (result) blended panorama image ([height x width x 3])
    """
    
    # student_code start
    result = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    for img, H_i in zip(images, H):
        H_total = T @ H_i

        warped_img = cv2.warpPerspective(img, H_total, (width, height))

        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        mask = (gray > 0).astype(np.uint8)

        # Compute distance transform for alpha blending
        distance = ndimage.distance_transform_edt(mask)
        alpha = distance / (np.max(distance) + 1e-9)

        # Expand alpha to three channels
        alpha_3 = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

        # Blend the warped image into the result
        result += warped_img * alpha_3
        weight_sum += alpha
    
    # Normalize
    weight_3 = np.repeat(weight_sum[:, :, np.newaxis], 3, axis=2)
    result = result / (weight_3 + 1e-9)

    result = np.clip(result, 0, 255).astype(np.uint8)
    # student_code end

    return result
