#####TUWIEN - CV: Task2 - Image Stitching
#####*********+++++++++*******++++ Group 14
from typing import List, Tuple
from numpy.linalg import inv
import numpy as np
import mapping
import random
import cv2


def get_geometric_transform(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Calculate a homography from the first set of points (p1) to the second (p2)

    Parameters
    ----------
    p1 : np.ndarray
        first set of points
    p2 : np.ndarray
        second set of points
    
    Returns
    ----------
    np.ndarray
        homography from p1 to p2
    """

    num_points = len(p1)
    A = np.zeros((2 * num_points, 9))
    for p in range(num_points):
        first = np.array([p1[p, 0], p1[p, 1], 1])
        A[2 * p] = np.concatenate(([0, 0, 0], -first, p2[p, 1] * first))
        A[2 * p + 1] = np.concatenate((first, [0, 0, 0], -p2[p, 0] * first))
    U, D, V = np.linalg.svd(A)
    H = V[8].reshape(3, 3)

    # homography from p1 to p2
    return (H / H[-1, -1]).astype(np.float32)


def get_transform(kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch]) -> Tuple[np.ndarray, List[int]]:
    """
    Estimate the homography between two set of keypoints by implementing the RANSAC algorithm
    HINT: random.sample(..), transforms.get_geometric_transform(..), cv2.perspectiveTransform(..)

    Parameters
    ----------
    kp1 : List[cv2.KeyPoint]
        keypoints left image ([number_of_keypoints] - KeyPoint)
    kp2 :  List[cv2.KeyPoint]
        keypoints right image ([number_of_keypoints] - KeyPoint)
    matches : List[cv2.DMatch]
        indices of matching keypoints ([number_of_matches] - DMatch)
    
    Returns
    ----------
    np.ndarray
        homographies from left (kp1) to right (kp2) image ([3 x 3] - float)
    List[int]
        inliers : list of indices, inliers in 'matches' ([number_of_inliers x 1] - int)
    """

    # student_code start
    N = 1000 # number of iterations
    T = 5.0 # threshold

    max_inliers = 0
    inliers = []
    trans = None

    all_kp1 = np.asarray([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    all_kp2 = np.asarray([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

    for _ in range(N):
        selected_matches = random.sample(matches, 4)
        # cv2.DMatch consits of queryIdx, trainIdx and distance
        # queryIdx = Index of KeyPoints in first Img/KeyPoint list, point is matched
        # trainIdx = Index of KeyPoints in second Img/KeyPoint list, corresponing (matched) point
        selected_kp1 = np.asarray([kp1[m.queryIdx].pt for m in selected_matches], dtype=np.float32)
        selected_kp2 = np.asarray([kp2[m.trainIdx].pt for m in selected_matches], dtype=np.float32)

        # Estimate homography between the selected points
        H = get_geometric_transform(selected_kp1, selected_kp2)
        # Transform points of matches in the first image
        all_kp1_trans = cv2.perspectiveTransform(all_kp1.reshape(-1,1,2), H) # needs (N, 1, 2)
        all_kp1_trans_2D = all_kp1_trans.reshape(-1,2) # in 2D zurückwandeln

        # Calculate euclidean distance between transformed points of img 1 and second img
        distances = np.linalg.norm(all_kp1_trans_2D - all_kp2, axis=1) # calculates for each zeile seperately
        # Determines number of inliers 
        mask_inliers = distances < T
        num_inliers = np.sum(mask_inliers)

        # Take the homography that has the max. number of inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            inliers = np.where(mask_inliers)[0] # indices of inliers
            trans = H

    # Re-estiamte the homography with ALL inliers one more time -> more accurate result
    inlier_kp1 = all_kp1[inliers]
    inlier_kp2 = all_kp2[inliers]
    trans = get_geometric_transform(inlier_kp1, inlier_kp2)
    
    # student_code end
    return trans, inliers


def to_center(desc: List[np.ndarray], kp: List[cv2.KeyPoint]) -> List[np.ndarray]:
    """
    Prepare all homographies by calculating the transforms from all other images
    to the reference image of the panorama (center image)
    First use mapping.calculate_matches(..) and get_transform(..) to get homographies between
    two consecutive images from left to right, then calculate and return the homographies to the center image
    HINT: inv(..), pay attention to the matrix multiplication order!!
    
    Parameters
    ----------
    desc : List[np.ndarray]
        list of descriptors ([number_of_images x num_of_keypoints, 128] - float)
    kp : List[cv2.KeyPoint]
        list of keypoints ([number_of_images x number_of_keypoints] - KeyPoint)
    
    Returns
    ----------
    List[np.ndarray]
        (H_center) list of homographies to the center image ( [number_of_images x 3 x 3] - float)
    """

    # student_code start
    raise NotImplementedError("TO DO in transforms.py")
    # student_code end

    return H_center


def get_panorama_extents(images: List[np.ndarray], H: List[np.ndarray]) -> Tuple[np.ndarray, int, int]:
    """
    Calculate the extent of the panorama by transforming the corners of every image
    and geht the minimum and maxima in x and y direction, as you read in the assignment description.
    Together with the panorama dimensions, return a translation matrix 'T' which transfers the
    panorama in a positive coordinate system. Remember that the origin of opencv images is in the upper left corner
    HINT: cv2.perspectiveTransform(..)

    Parameters
    ----------
    images : List[np.ndarray]
        list of images
    H : List[np.ndarray]
        list of homographies to center image ([number_of_images x 3 x 3])

    Returns
    ---------
    np.ndarray
        T : transformation matrix to translate the panorama to positive coordinates ([3 x 3])
    int
        width of panorama (in pixel)
    int
        height of panorama (in pixel)
    """

    # student_code start
    raise NotImplementedError("TO DO in transforms.py")
    # student_code end

    return T, width, height
