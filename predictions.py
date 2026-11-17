
import os
import gc
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from pathlib import Path

# Model configuration
ENCODER = 'resnet34'
ENCODER_WEIGHTS = None
CLASSES = 1
ACTIVATION = 'sigmoid'

DEVICE = torch.device('cpu')
torch.set_num_threads(1)

# Load model only once
checkpoint_path = './model/best_model.pth'

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True
)

model = smp.Unet(
    encoder_name=checkpoint['encoder'],
    encoder_weights=None,
    classes=checkpoint['classes'],
    activation='sigmoid',
    decoder_attention_type=checkpoint['attention']
)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

del checkpoint
gc.collect()

# Preprocessing
def preprocess_image(image):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER, 'imagenet'
    )

    image = cv2.resize(image, (256, 256))
    image = preprocessing_fn(image)
    image = image.transpose(2, 0, 1).astype('float32')

    return image


def predict(image_url, filename):


    # Read uploaded image
    input_image = cv2.imread(image_url)

    if input_image is None:
        raise ValueError("Unable to read uploaded image")

    input_image = cv2.cvtColor(
        input_image,
        cv2.COLOR_BGR2RGB
    )

    # Save original image
    output_path1 = (
        'static/output/' + Path(filename).stem + '.png'
    )

    os.makedirs('static/output', exist_ok=True)
    os.makedirs('static/res', exist_ok=True)

    cv2.imwrite(
        output_path1,
        cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    )

    # Preprocess image
    image = preprocess_image(input_image)

    image_tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():
        output = model(image_tensor)

    prediction = output.squeeze().cpu().numpy()

    # Generate road mask
    road_mask = (prediction > 0.5).astype(np.uint8) * 255

    # Resize mask to original image size
    road_mask = cv2.resize(
        road_mask,
        (input_image.shape[1], input_image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Save predicted road mask
    output_path2 = (
        'static/res/' + Path(filename).stem + '.png'
    )

    cv2.imwrite(output_path2, road_mask)
    del image_tensor, output, prediction, road_mask
    gc.collect()
    
    return output_path1, output_path2
