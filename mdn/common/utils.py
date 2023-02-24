import random
import numpy as np
import os


def set_seeds(seed=1337):

    random.seed(seed)
    np.random.seed(seed)


def get_pose_path(base_path, speaker, pose_fn):
    return os.path.join(base_path, speaker, 'keypoints_simple', pose_fn)


def get_frame_path(base_path, speaker, frame_fn):
    return os.path.join(base_path, speaker, 'frames', frame_fn)


def get_video_path(base_path, speaker, video_fn):
    return os.path.join(base_path, speaker, 'videos', video_fn)


def get_face_path(base_path, speaker, pose_fn):
    return os.path.join(base_path, speaker, 'keypoints_all', pose_fn)


def delete_face_keypoints(k, axis=2):
    '''
    Deletes the two eyes and nose from a model_23 set of openpose keypoints

    :param k: one set of keypoints with shape [2, #num_keypoints]
    '''
    return np.delete(k, [7, 8, 9], axis=axis)  # in model_23 of openpose 7 is nose, 8,9 are eyes