import matplotlib.pyplot as plt
from common.consts import *
import cv2
import numpy as np
import pandas as pd
from utils import delete_face_keypoints


def draw_curve(img, shape, idx_list, color=(0, 255, 0), lineWidth=LINE_WIDTH_CONST):
    for i in range(len(idx_list) - 1):
        cv2.line(img, (shape[idx_list[i], 0], shape[idx_list[i], 1]), (shape[idx_list[i + 1], 0], shape[idx_list[i + 1], 1]), color, lineWidth)


def plot_mouth(img, keypoints, line_width=LINE_WIDTH_CONST):
    draw_curve(img, keypoints, Mouth, lineWidth=line_width)


def plot_nose(img, keypoints, line_width=LINE_WIDTH_CONST):
    draw_curve(img, keypoints, Nose, lineWidth=line_width)


def plot_leftBrow(img, keypoints, line_width=LINE_WIDTH_CONST):
    draw_curve(img, keypoints, leftBrow, lineWidth=line_width)


def plot_rightBrow(img, keypoints, line_width=LINE_WIDTH_CONST):
    draw_curve(img, keypoints, rightBrow, lineWidth=line_width)


def plot_leftEye(img, keypoints, line_width=LINE_WIDTH_CONST):
    draw_curve(img, keypoints, leftEye, lineWidth=line_width)


def plot_rightEye(img, keypoints, line_width=LINE_WIDTH_CONST):
    draw_curve(img, keypoints, rightEye, lineWidth=line_width)


def plot_other(img, keypoints, line_width=LINE_WIDTH_CONST):
    draw_curve(img, keypoints, other, lineWidth=line_width)


def plot_face(img, keypoints, line_width=LINE_WIDTH_CONST):
    plot_mouth(img, keypoints, line_width)
    plot_nose(img, keypoints, line_width)
    plot_leftBrow(img, keypoints, line_width)
    plot_rightBrow(img, keypoints, line_width)
    plot_leftEye(img, keypoints, line_width)
    plot_rightEye(img, keypoints, line_width)
    plot_other(img, keypoints, line_width)


def plot_body_right_keypoints(img, keypoints, line_width=LINE_WIDTH_CONST):
    _keypoints = np.array(BASE_KEYPOINT + RIGHT_BODY_KEYPOINTS)
    draw_curve(img, keypoints, _keypoints, lineWidth=line_width)


def plot_body_left_keypoints(img, keypoints, line_width=LINE_WIDTH_CONST):
    _keypoints = np.array(BASE_KEYPOINT + LEFT_BODY_KEYPOINTS)
    draw_curve(img, keypoints, _keypoints, lineWidth=line_width)


def plot_left_hand_keypoints(img, keypoints, line_width=LINE_WIDTH_CONST):
    for i in range(5):
        _keypoints = np.array(LEFT_HAND_KEYPOINTS(i))
        draw_curve(img, keypoints, _keypoints, lineWidth=line_width)


def plot_right_hand_keypoints(img, keypoints, line_width=LINE_WIDTH_CONST):
    for i in range(5):
        _keypoints = np.array(RIGHT_HAND_KEYPOINTS(i))
        draw_curve(img, keypoints, _keypoints, lineWidth=line_width)

def plot_gesture(img, keypoints, line_width=LINE_WIDTH_CONST):
    plot_body_right_keypoints(img, keypoints, line_width)
    plot_body_left_keypoints(img, keypoints, line_width)
    plot_left_hand_keypoints(img, keypoints, line_width)
    plot_right_hand_keypoints(img, keypoints, line_width)

#######################################################################################


def vis_landmark_on_img(img, shape_face=None, shape_pose=None, line_width=LINE_WIDTH_CONST, path=None):
    if shape_face is not None:
        plot_face(img, shape_face,line_width)
    if shape_pose is not None:
        plot_gesture(img, shape_pose, line_width)
    if path is None:
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(path, img)

