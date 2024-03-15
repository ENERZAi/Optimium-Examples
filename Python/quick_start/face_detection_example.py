import numpy as np
import numpy.typing as npt
from pathlib import Path
from PIL import Image
from typing import List, Optional, Union

from fdlite.types import Detection, Rect
from fdlite.nms import non_maximum_suppression
from fdlite.transform import detection_letterbox_removal, image_to_tensor
from fdlite.render import Colors, detections_to_render_data, render_to_image

from util import (
    SSD_OPTIONS_SHORT,
    _decode_boxes,
    _get_sigmoid_scores,
    _convert_to_detections,
    _ssd_generate_anchors
)

import optimium.runtime as rt


# threshold for confidence scores
MIN_SCORE = 0.5
# NMS similarity threshold
MIN_SUPPRESSION_THRESHOLD = 0.3
    

def preprocess_image(
    height,
    width,
    image: Union[type(Image), np.ndarray, str],
    roi: Optional[Rect] = None,
):
    return image_to_tensor(
        image,
        roi,
        output_size=(width, height),
        keep_aspect_ratio=True,
        output_range=(-1, 1))

    
# THIS IS WHERE OPTIMIUM RUNTIME EXECUTES MODEL
def optimium_infer(
    input_image: Image, 
    model_path: Path
):
    # Initiate Optimium Runtime context 
    context = rt.Context()
    
    # Load optimized model (this example assumes inferencing Face Detection model)
    model = context.load_model(model_path)
    
    # Get input and output information
    input = model.get_input_tensor_info
    output = model.get_output_tensor_info
    input_shape = model.get_input_tensor_info(0).shape  # Face Detection has only one input (NHWC)
    height, width = input_shape[1:3]
    
    # Preprocess input image
    image_data = preprocess_image(
        height,
        width,
        input_image
    )
    input_data = image_data.tensor_data[np.newaxis]
    
    # Create inference request and set input
    request = model.create_request()
    request.set_inputs(input_data)
    
    # Perform inference
    request.infer()
    request.wait()
    
    # Inference result
    output_data = request.get_outputs()  # This will be a list of numpy.ndarray
    
    # For post-processing
    info_for_postprocess = (
        input_shape,
        image_data,                        
    )
    
    return info_for_postprocess, output_data


# Post-processing for Face Detection model
def post_process_inference_to_dection(
    info_for_postprocess,
    output_data: List[npt.NDArray]
) -> List[Detection]:
    input_shape, image_data = info_for_postprocess
    anchors = _ssd_generate_anchors(SSD_OPTIONS_SHORT)
    
    raw_boxes = output_data[0]
    raw_scores = output_data[1]
    
    boxes = _decode_boxes(input_shape, anchors, raw_boxes)
    scores = _get_sigmoid_scores(raw_scores)
    
    detections = _convert_to_detections(boxes, scores)
    pruned_detections = non_maximum_suppression(
                            detections,
                            MIN_SUPPRESSION_THRESHOLD, 
                            MIN_SCORE,
                            weighted=True)
    detections = detection_letterbox_removal(
        pruned_detections, image_data.padding)
    return detections
    
    
def face_detection(image_path: Path, model_path: Path, output_path: Path):
    image = Image.open(image_path).rotate(-90.0, expand=True)
    info_for_postprocess, output_data = optimium_infer(image, model_path)
    detected_faces = post_process_inference_to_dection(info_for_postprocess, output_data)
    
    if not len(detected_faces):
        print('no faces detected :(')
    else:
        render_data = detections_to_render_data(detected_faces, bounds_color=Colors.GREEN, line_width=5)
        render_to_image(render_data, image).save(output_path)


if __name__ == "__main__":
    model_path = Path("optimium_model_output")
    image_path = Path("sample_image.jpg")
    output_path = Path("sample_face_detection.jpg")
    face_detection(
        image_path=image_path,
        model_path=model_path,
        output_path=output_path
    )
