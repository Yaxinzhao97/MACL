# keypoints consts
BASE_KEYPOINT = [0]
RIGHT_BODY_KEYPOINTS = [1, 2, 3, 28]
LEFT_BODY_KEYPOINTS = [4, 5, 6, 7]
LEFT_HAND_KEYPOINTS = lambda x: [7] + [8 + (x * 4) + j for j in range(4)]
RIGHT_HAND_KEYPOINTS = lambda x: [28] + [29 + (x * 4) + j for j in range(4)]
ALL_RIGHT_HAND_KEYPOINTS = [3] + list(range(31, 52))
ALL_LEFT_HAND_KEYPOINTS = [6] + list(range(10, 31))
BODY_KEYPOINTS = RIGHT_BODY_KEYPOINTS + LEFT_BODY_KEYPOINTS

# training consts
SR = 16000
FRAMES_PER_SAMPLE = 64
AUDIO_SAMPLE_SHAPE = (FRAMES_PER_SAMPLE, 32)
POSE_SAMPLE_SHAPE = (FRAMES_PER_SAMPLE, 98)
AUDIO_SHAPE = 67267
FPS = 30./2
FRAMES_NUM = 263

# plotting consts
LINE_WIDTH_CONST = 1
FRAME_X_ANCHOR_POINT = 1000.
FRAME_X_WIDTH = 2000.
FRAME_Y_MIN = -300.
FRAME_Y_MAX = 1000.
FRAME_Y_HEIGHT = FRAME_Y_MAX - FRAME_Y_MIN

# time consts
MILLISECOND = 1. / 1000000
SECOND = 1.
MINUTE = 60. * SECOND
HOUR = 60. * MINUTE

Mouth = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]

Nose = [27, 28, 29, 30, 31, 32, 33, 34, 35]

leftBrow = [17, 18, 19, 20, 21]
rightBrow = [22, 23, 24, 25, 26]

leftEye = [36, 37, 38, 39, 40, 41]
rightEye = [42, 43, 44, 45, 46, 47]

other = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]