#####TUWIEN - CV: Task2 - Image Stitching
#####*********+++++++++*******++++ Group 14
from typing import List, Tuple
from matplotlib.pyplot import angle_spectrum
import numpy as np
# from cyvlfeat.sift.sift import sift
# import cyvlfeat
import os
import cv2
import glob


def get_panorama_data(path: str) -> Tuple[List[np.ndarray], List[cv2.KeyPoint], List[np.ndarray]]:
    """
    Loop through images in given folder (do not forget to sort them), extract SIFT points
    and return images, keypoints and descriptors
    This time we need to work with color images. Since OpenCV uses BGR you need to swap the channels to RGB
    HINT: sorted(..), glob.glob(..), cv2.imread(..), sift=cv2.SIFT_create(), sift.detectAndCompute(..)

    Parameters
    ----------
    path : str
        path to image folder

    Returns
    ---------
    List[np.ndarray]
        img_data : list of images
    List[cv2.Keypoint]
        all_keypoints : list of keypoints ([number_of_images x number_of_keypoints] - KeyPoint)
    List[np.ndarray]
        all_descriptors : list of descriptors ([number_of_images x num_of_keypoints, 128] - float)
    """
    
    # student_code start
    img_data = list()
    all_keypoints = list()
    all_descriptors = list()

    image_paths = sorted(glob.glob(os.path.join(path, '*.jpg')))

    # SIFT detects distinctive keypoints in imgs that are invariant to scale and rotation,
    # computes 128-dimensional descriptors for each and can match these descriptors between imgs.
    sift = cv2.SIFT_create()

    for p in image_paths:

        img_bgr = cv2.imread(p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Since SIFT is based on intensities (not colors), grayscaled imgs are sufficient.
        # However, 3-channel imgs could also be used as a parameter because of the 
        # internal grayscale conversion (see OpenCV Tutorial of SIFT).
        kp, des = sift.detectAndCompute(img_gray, None)

        # keypoints: distinctive positions in the image, e.g. corners, that are stable under scale and rotation changes
        # descriptors: 128-dims feature vectors describing the local appearance around each
        img_data.append(img_rgb)
        all_keypoints.append(kp) 
        all_descriptors.append(des) # (num of kp) x 128 vectors
    # student_code end

    return img_data, all_keypoints, all_descriptors
