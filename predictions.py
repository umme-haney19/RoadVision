
import os
import gc
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from pathlib import Path

# Model configuration
ENCODER = 'resnet34'
DEVICE = torch.device('cpu')

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled = False

# Load model only once
checkpoint_path = './model/best_model.pth'

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True,
    mmap=True
)

model = smp.Unet(
    encoder_name=checkpoint['encoder'],
    encoder_weights=None,
    classes=checkpoint['classes'],
    activation='sigmoid',
    decoder_attention_type=checkpoint['attention']
)

model.load_state_dict(checkpoint['model_state_dict'], assign=True)
model.to(DEVICE)
model.eval()

del checkpoint
gc.collect()


# Preprocessing
def preprocess_image(image):

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER, 'imagenet'
    )

    image = preprocessing_fn(image)
    image = image.transpose(2, 0, 1).astype('float32')

    return image


# Prediction
def predict(image_url, filename):

    # Read uploaded image
    input_image = cv2.imread(image_url)

    if input_image is None:
        raise ValueError("Unable to read uploaded image")

    # Store original dimensions
    original_height, original_width = input_image.shape[:2]

    # Create output folders
    os.makedirs('static/output', exist_ok=True)
    os.makedirs('static/res', exist_ok=True)

    # Save original image
    output_path1 = (
        'static/output/' + Path(filename).stem + '.png'
    )

    cv2.imwrite(output_path1, input_image)

    # Resize before preprocessing
    small_image = cv2.resize(
        input_image,
        (256, 256),
        interpolation=cv2.INTER_AREA
    )

    small_image = cv2.cvtColor(
        small_image,
        cv2.COLOR_BGR2RGB
    )

    # Preprocess image
    image = preprocess_image(small_image)

    del input_image, small_image
    gc.collect()

    # Create tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0)

    del image
    gc.collect()

    # Run prediction
    with torch.inference_mode():
        output = model(image_tensor)

    # Generate road mask
    prediction = output.squeeze().cpu().numpy()

    road_mask = (
        (prediction > 0.5).astype(np.uint8) * 255
    )

    # Resize mask to original image dimensions
    road_mask = cv2.resize(
        road_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )

    # Save predicted road mask
    output_path2 = (
        'static/res/' + Path(filename).stem + '.png'
    )

    cv2.imwrite(output_path2, road_mask)

    # Release memory
    del image_tensor, output, prediction, road_mask
    gc.collect()

    return output_path1, output_path2
