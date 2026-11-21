
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

import gc
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ONNX Runtime configuration
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

session = ort.InferenceSession(
    "roadvision.onnx",
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# ResNet34 ImageNet preprocessing
def preprocess_image(image):
    image = image.astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )
    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    image = (image - mean) / std
    image = image.transpose(2, 0, 1)

    return np.ascontiguousarray(image, dtype=np.float32)


def predict(image_url, filename):

    input_image = cv2.imread(
        str(image_url),
        cv2.IMREAD_REDUCED_COLOR_8
    )

    if input_image is None:
        raise ValueError("Unable to read uploaded image")

    original_height, original_width = input_image.shape[:2]

    os.makedirs("static/output", exist_ok=True)
    os.makedirs("static/res", exist_ok=True)

    stem = Path(filename).stem

    output_path1 = f"static/output/{stem}.png"
    output_path2 = f"static/res/{stem}.png"

    cv2.imwrite(output_path1, input_image)

    small_image = cv2.resize(
        input_image,
        (64, 64),
        interpolation=cv2.INTER_AREA
    )

    small_image = cv2.cvtColor(
        small_image,
        cv2.COLOR_BGR2RGB
    )

    image = preprocess_image(small_image)

    image_tensor = np.expand_dims(image, axis=0)

    output = session.run(
        None,
        {input_name: image_tensor}
    )[0]

    prediction = output.squeeze()

    road_mask = (
        (prediction > 0.5).astype(np.uint8) * 255
    )

    road_mask = cv2.resize(
        road_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imwrite(output_path2, road_mask)

    del input_image, small_image
    del image, image_tensor, output
    del prediction, road_mask

    gc.collect()

    return output_path1, output_path2
