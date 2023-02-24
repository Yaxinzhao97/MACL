import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import cv2
import sys
sys.path.append('..')
from vis import plot_face

t_shape_idx = (27, 28, 29, 30, 33, 36, 39, 42, 45)


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # Step added for scalar (deprecated)
    p_deno = np.sum(AA**2, axis=0)
    y_nume = np.sum(BB**2, axis=0)
    s = np.identity(m+1)
    s[:m, :m] = s[:m, :m] * (y_nume / p_deno) ** 0.25

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    # Step : (Deprecated for Scalar)
    # T = np.dot(s, T)

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=50, tolerance=0.0001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):

        distances = np.sum((src[:m, :] - dst[:m, :])**2)
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, :].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i


def single_landmark_2d_register(fl2d, anchor_t_shape, display=False):
    """
    Register a single 3d landmark file
    """
    # Step 1 : Load and Smooth
    from scipy.signal import savgol_filter
    # fl3d (b=64, 140)
    lines = savgol_filter(fl2d, 7, 3, axis=0)
    #
    all_landmarks = lines.reshape((-1, 70, 2))
    w, h = int(np.max(all_landmarks[:, :, 0])) + 20, int(np.max(all_landmarks[:, :, 1])) + 20
    #
    # # Step 2 : setup anchor face

    anchor_t_shape = anchor_t_shape[t_shape_idx, :]
    registered_landmarks_to_save = []
    registered_affine_mat_to_save = []
    # for each line
    for line in lines:

        landmarks = line.reshape(70, 2)

        # Step 3 : ICP on (frame, anchor)
        frame_t_shape = landmarks[t_shape_idx, :]

        T, distance, itr = icp(frame_t_shape, anchor_t_shape)

        # Step 4 : Affine transform
        landmarks = np.hstack((landmarks, np.ones((70, 1))))
        registered_landmarks = np.dot(T, landmarks.T).T
        err = np.mean(np.sqrt(np.sum((registered_landmarks[t_shape_idx, 0:2] - anchor_t_shape) ** 2, axis=1)))
        # print(err, distance, itr)

        # Step 5 : Save is requested
        registered_landmarks_to_save.append(registered_landmarks[:, 0:2][np.newaxis, :])
        registered_affine_mat_to_save.append(T[np.newaxis, :])

        # Step 5.5 (optional) : visualize ori / registered faces (Isolated in Black BG)
        if (display):
            img = np.zeros((h, w * 2, 3), np.uint8)
            plot_face(img, landmarks.astype(np.int))
            registered_landmarks[:, 0] += w
            plot_face(img, registered_landmarks.astype(np.int))
            cv2.imshow('img', img)
            # if (cv2.waitKey(30) == ord('q')):
            #     break
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    registered_landmarks_to_save = np.concatenate(registered_landmarks_to_save, axis=0)
    registered_affine_mat_to_save = np.concatenate(registered_affine_mat_to_save, axis=0)
    return registered_landmarks_to_save, registered_affine_mat_to_save



