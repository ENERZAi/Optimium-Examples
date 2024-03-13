import numpy as np
from typing import List

from fdlite.types import Detection
from fdlite.transform import sigmoid

# score limit is 100 in mediapipe and leads to overflows with IEEE 754 floats
# this lower limit is safe for use with the sigmoid functions and float32
RAW_SCORE_LIMIT = 80
# threshold for confidence scores
MIN_SCORE = 0.5


# (reference: modules/face_detection/face_detection_short_range_common.pbtxt)
SSD_OPTIONS_SHORT = {
    'num_layers': 4,
    'input_size_height': 128,
    'input_size_width': 128,
    'anchor_offset_x': 0.5,
    'anchor_offset_y': 0.5,
    'strides': [8, 16, 16, 16],
    'interpolated_scale_aspect_ratio': 1.0
}


def _decode_boxes(input_shape, anchors, raw_boxes: np.ndarray) -> np.ndarray:
    """Simplified version of
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    """
    # width == height so scale is the same across the board
    scale = input_shape[1]
    num_points = raw_boxes.shape[-1] // 2
    # scale all values (applies to positions, width, and height alike)
    boxes = raw_boxes.reshape(-1, num_points, 2) / scale
    # adjust center coordinates and key points to anchor positions
    boxes[:, 0] += anchors
    for i in range(2, num_points):
        boxes[:, i] += anchors
    # convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
    center = np.array(boxes[:, 0])
    half_size = boxes[:, 1] / 2
    boxes[:, 0] = center - half_size
    boxes[:, 1] = center + half_size
    return boxes


def _get_sigmoid_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Extracted loop from ProcessCPU (line 327) in
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    """
    # just a single class ("face"), which simplifies this a lot
    # 1) thresholding; adjusted from 100 to 80, since sigmoid of [-]100
    #    causes overflow with IEEE single precision floats (max ~10e38)
    raw_scores[raw_scores < -RAW_SCORE_LIMIT] = -RAW_SCORE_LIMIT
    raw_scores[raw_scores > RAW_SCORE_LIMIT] = RAW_SCORE_LIMIT
    # 2) apply sigmoid function on clipped confidence scores
    return sigmoid(raw_scores)


def _convert_to_detections(
        boxes: np.ndarray,
        scores: np.ndarray
    ) -> List[Detection]:
        """Apply detection threshold, filter invalid boxes and return
        detection instance.
        """
        # return whether width and height are positive
        def is_valid(box: np.ndarray) -> bool:
            return np.all(box[1] > box[0])

        score_above_threshold = scores > MIN_SCORE
        filtered_boxes = boxes[np.argwhere(score_above_threshold)[:, 1], :]
        filtered_scores = scores[score_above_threshold]
        return [Detection(box, score)
                for box, score in zip(filtered_boxes, filtered_scores)
                if is_valid(box)]


def _ssd_generate_anchors(opts: dict) -> np.ndarray:
    """This is a trimmed down version of the C++ code; all irrelevant parts
    have been removed.
    (reference: mediapipe/calculators/tflite/ssd_anchors_calculator.cc)
    """
    layer_id = 0
    num_layers = opts['num_layers']
    strides = opts['strides']
    assert len(strides) == num_layers
    input_height = opts['input_size_height']
    input_width = opts['input_size_width']
    anchor_offset_x = opts['anchor_offset_x']
    anchor_offset_y = opts['anchor_offset_y']
    interpolated_scale_aspect_ratio = opts['interpolated_scale_aspect_ratio']
    anchors = []
    while layer_id < num_layers:
        last_same_stride_layer = layer_id
        repeats = 0
        while (last_same_stride_layer < num_layers and
               strides[last_same_stride_layer] == strides[layer_id]):
            last_same_stride_layer += 1
            # aspect_ratios are added twice per iteration
            repeats += 2 if interpolated_scale_aspect_ratio == 1.0 else 1
        stride = strides[layer_id]
        feature_map_height = input_height // stride
        feature_map_width = input_width // stride
        for y in range(feature_map_height):
            y_center = (y + anchor_offset_y) / feature_map_height
            for x in range(feature_map_width):
                x_center = (x + anchor_offset_x) / feature_map_width
                for _ in range(repeats):
                    anchors.append((x_center, y_center))
        layer_id = last_same_stride_layer
    return np.array(anchors, dtype=np.float32)