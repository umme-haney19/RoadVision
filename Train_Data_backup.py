import os
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp
import albumentations as album

warnings.filterwarnings("ignore")


# =========================================================
# 1. CONFIGURATION
# =========================================================

DATA_DIR = Path("./Datasets/DeepGlobe")

IMAGE_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-5

# Set to None to use the complete available training split.
# CPU training can be slow, so this keeps the first run manageable.
MAX_TRAIN_IMAGES = 500
MAX_VALID_IMAGES = 100

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
ATTENTION = "scse"

DEVICE = torch.device("cpu")
torch.set_num_threads(4)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MODEL_DIR = Path("./model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.pth"

print("Training device:", DEVICE)


# =========================================================
# 2. CHECK DATASET FILES
# =========================================================

metadata_path = DATA_DIR / "metadata.csv"
class_dict_path = DATA_DIR / "class_dict.csv"

if not metadata_path.exists():
    raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

if not class_dict_path.exists():
    raise FileNotFoundError(f"class_dict.csv not found: {class_dict_path}")

metadata_df = pd.read_csv(metadata_path)

required_columns = {"image_id", "sat_image_path", "mask_path", "split"}
missing_columns = required_columns - set(metadata_df.columns)

if missing_columns:
    raise ValueError(f"Missing columns in metadata.csv: {missing_columns}")

metadata_df = metadata_df[
    metadata_df["split"].astype(str).str.lower() == "train"
].copy()

if metadata_df.empty:
    raise ValueError("No rows with split='train' were found in metadata.csv.")

metadata_df = metadata_df[
    ["image_id", "sat_image_path", "mask_path"]
].copy()


def resolve_dataset_path(value):
    """Resolve a metadata path relative to the dataset directory."""
    path = Path(str(value))

    if path.is_absolute():
        return str(path)

    return str(DATA_DIR / path)


metadata_df["sat_image_path"] = metadata_df["sat_image_path"].apply(
    resolve_dataset_path
)
metadata_df["mask_path"] = metadata_df["mask_path"].apply(
    resolve_dataset_path
)

# Remove rows whose image or mask files are missing.
valid_paths = (
    metadata_df["sat_image_path"].apply(lambda p: Path(p).is_file())
    & metadata_df["mask_path"].apply(lambda p: Path(p).is_file())
)

missing_count = int((~valid_paths).sum())
if missing_count:
    print(f"Warning: Skipping {missing_count} rows with missing image/mask files.")

metadata_df = metadata_df[valid_paths].reset_index(drop=True)

if len(metadata_df) < 2:
    raise ValueError(
        "Not enough valid image/mask pairs. Check metadata.csv and dataset folders."
    )

metadata_df = metadata_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# =========================================================
# 3. TRAIN / VALIDATION SPLIT
# =========================================================

valid_count = max(1, int(round(len(metadata_df) * 0.1)))
valid_count = min(valid_count, len(metadata_df) - 1)

valid_df = metadata_df.iloc[:valid_count].copy()
train_df = metadata_df.iloc[valid_count:].copy()

if MAX_TRAIN_IMAGES is not None:
    train_df = train_df.sample(
        n=min(MAX_TRAIN_IMAGES, len(train_df)),
        random_state=42
    )

if MAX_VALID_IMAGES is not None:
    valid_df = valid_df.sample(
        n=min(MAX_VALID_IMAGES, len(valid_df)),
        random_state=42
    )

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

print("Training images:", len(train_df))
print("Validation images:", len(valid_df))


# =========================================================
# 4. FIND ROAD CLASS RGB COLOUR
# =========================================================

class_dict = pd.read_csv(class_dict_path)

required_class_columns = {"name", "r", "g", "b"}
missing_class_columns = required_class_columns - set(class_dict.columns)

if missing_class_columns:
    raise ValueError(
        f"Missing columns in class_dict.csv: {missing_class_columns}"
    )

road_row = class_dict[
    class_dict["name"].astype(str).str.strip().str.lower() == "road"
]

if road_row.empty:
    raise ValueError("Road class was not found in class_dict.csv.")

ROAD_RGB = road_row[["r", "g", "b"]].iloc[0].to_numpy(dtype=np.uint8)

print("Expected road RGB:", ROAD_RGB)


# =========================================================
# 5. VISUALIZATION FUNCTION
# =========================================================

def visualize(**images):
    number_of_images = len(images)

    plt.figure(figsize=(6 * number_of_images, 6))

    for index, (name, image) in enumerate(images.items(), start=1):
        plt.subplot(1, number_of_images, index)

        if image.ndim == 2:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)

        plt.title(name.replace("_", " ").title())
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# 6. DATA AUGMENTATION
# =========================================================

def get_training_augmentation():
    return album.Compose([
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
    ])


# =========================================================
# 7. PREPROCESSING
# =========================================================

preprocessing_fn = smp.encoders.get_preprocessing_fn(
    ENCODER,
    ENCODER_WEIGHTS
)


def preprocess_image(image):
    """Apply the same encoder preprocessing used during training."""
    return preprocessing_fn(image)


def image_to_tensor(image, **kwargs):
    return image.transpose(2, 0, 1).astype(np.float32)


def mask_to_tensor(mask, **kwargs):
    return np.expand_dims(mask.astype(np.float32), axis=0)


def get_preprocessing():
    return None


# =========================================================
# 8. CUSTOM ROAD DATASET
# =========================================================

class RoadsDataset(Dataset):

    def __init__(self, dataframe, augmentation=None, preprocessing=None):
        self.image_paths = dataframe["sat_image_path"].tolist()
        self.mask_paths = dataframe["mask_path"].tolist()

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Image and mask counts do not match.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        # OpenCV loads BGR. Convert both to RGB before comparing colours.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Resize image and mask.
        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        # Road pixels = 1; all other classes = 0.
        binary_mask = np.all(mask == ROAD_RGB, axis=-1).astype(np.float32)

        # Apply augmentation to image and mask together.
        if self.augmentation is not None:
            sample = self.augmentation(
                image=image,
                mask=binary_mask
            )
            image = sample["image"]
            binary_mask = sample["mask"]

        # Apply encoder preprocessing
        image = preprocessing_fn(image)
            
        # Convert image to PyTorch format
        image = image.transpose(2, 0, 1).astype(np.float32)
            
        # Convert mask to PyTorch format
        binary_mask = np.expand_dims(
            binary_mask.astype(np.float32),
            axis=0
        )

        return image, binary_mask


# =========================================================
# 9. CREATE DATASETS
# =========================================================

train_dataset = RoadsDataset(
    train_df,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing()
)

valid_dataset = RoadsDataset(
    valid_df,
    preprocessing=get_preprocessing()
)

print("Training dataset ready!")
print("Validation dataset ready!")


# =========================================================
# 10. CHECK AND VISUALIZE A SAMPLE MASK
# =========================================================

# Inspect the original mask colours before training.
sample_raw_mask = cv2.imread(
    train_df.iloc[0]["mask_path"],
    cv2.IMREAD_COLOR
)

if sample_raw_mask is None:
    raise FileNotFoundError("Could not read the first training mask.")

sample_raw_mask = cv2.cvtColor(sample_raw_mask, cv2.COLOR_BGR2RGB)
sample_raw_mask = cv2.resize(
    sample_raw_mask,
    (IMAGE_SIZE, IMAGE_SIZE),
    interpolation=cv2.INTER_NEAREST
)

colors, counts = np.unique(
    sample_raw_mask.reshape(-1, 3),
    axis=0,
    return_counts=True
)

print("\nActual RGB colours in first mask (showing up to 20):")
for color, count in zip(colors[:20], counts[:20]):
    print(color, ":", count)

print("\nExpected road RGB:", ROAD_RGB)

sample_image, sample_mask = train_dataset[0]

# Read a clean original image for display.
display_image = cv2.imread(train_df.iloc[0]["sat_image_path"])
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
display_image = cv2.resize(
    display_image,
    (IMAGE_SIZE, IMAGE_SIZE)
)

visualize(
    original_image=display_image,
    road_mask=sample_mask[0]
)


# =========================================================
# 11. DATA LOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

if len(train_loader) == 0 or len(valid_loader) == 0:
    raise ValueError("Training or validation DataLoader is empty.")


# =========================================================
# 12. ATTENTION U-NET MODEL
# =========================================================

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=1,
    activation=None,
    decoder_attention_type=ATTENTION
)

model = model.to(DEVICE)

print("\nAttention-Assisted U-Net created successfully!")


# =========================================================
# 13. LOSS FUNCTION: DICE + BCE
# =========================================================

class DiceBCELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(logits, targets)

        probabilities = torch.sigmoid(logits)

        smooth = 1.0

        intersection = (probabilities * targets).sum()

        dice_score = (
            (2.0 * intersection + smooth)
            / (probabilities.sum() + targets.sum() + smooth)
        )

        dice_loss = 1.0 - dice_score

        return bce_loss + dice_loss


loss_fn = DiceBCELoss()


# =========================================================
# 14. OPTIMIZER
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# 15. IOU METRIC
# =========================================================

def calculate_iou(logits, targets):

    predictions = torch.sigmoid(logits) > 0.5
    targets = targets > 0.5

    intersection = (predictions & targets).sum().item()
    union = (predictions | targets).sum().item()

    return intersection, union


# =========================================================
# 16. TRAINING FUNCTION
# =========================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_intersection = 0
    total_union = 0

    for images, masks in train_loader:

        images = images.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE, dtype=torch.float32)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        intersection, union = calculate_iou(
            outputs.detach(),
            masks
        )

        total_intersection += intersection
        total_union += union

    average_loss = total_loss / len(train_loader)

    iou = (
        total_intersection / total_union
        if total_union > 0 else 0.0
    )

    return average_loss, iou


# =========================================================
# 17. VALIDATION FUNCTION
# =========================================================

def validate_one_epoch():

    model.eval()

    total_loss = 0.0
    total_intersection = 0
    total_union = 0

    with torch.no_grad():

        for images, masks in valid_loader:

            images = images.to(DEVICE, dtype=torch.float32)
            masks = masks.to(DEVICE, dtype=torch.float32)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            total_loss += loss.item()

            intersection, union = calculate_iou(
                outputs,
                masks
            )

            total_intersection += intersection
            total_union += union

    average_loss = total_loss / len(valid_loader)

    iou = (
        total_intersection / total_union
        if total_union > 0 else 0.0
    )

    return average_loss, iou


# =========================================================
# 18. MODEL TRAINING
# =========================================================

if __name__ == "__main__":

    best_iou_score = -1.0

    train_losses = []
    valid_losses = []

    print("\nStarting Attention U-Net Training...")

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_iou = train_one_epoch()
        valid_loss, valid_iou = validate_one_epoch()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print("Training Loss:", round(train_loss, 4))
        print("Training IoU:", round(train_iou, 4))
        print("Validation Loss:", round(valid_loss, 4))
        print("Validation IoU:", round(valid_iou, 4))

        if valid_iou > best_iou_score:

            best_iou_score = valid_iou

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "encoder": ENCODER,
                "classes": 1,
                "attention": ATTENTION,
                "image_size": IMAGE_SIZE,
                "encoder_weights": ENCODER_WEIGHTS,
            }

            torch.save(checkpoint, MODEL_PATH)

            print(f"Best model saved successfully: {MODEL_PATH}")

    print("\nTraining completed!")
    print("Best Validation IoU:", best_iou_score)


    # =====================================================
    # 19. LOAD BEST MODEL
    # =====================================================

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model checkpoint was not created: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Best model loaded successfully!")


    # =====================================================
    # 20. GENERATE AND SAVE SAMPLE PREDICTIONS
    # =====================================================

    sample_preds_folder = Path("./sample_predictions")
    sample_preds_folder.mkdir(parents=True, exist_ok=True)

    print("\nGenerating road predictions...")

    for index in range(min(5, len(valid_dataset))):

        image_tensor, gt_mask = valid_dataset[index]

        x_tensor = torch.from_numpy(
            image_tensor
        ).unsqueeze(0).to(DEVICE, dtype=torch.float32)

        with torch.no_grad():
            logits = model(x_tensor)
            probability = torch.sigmoid(logits)

        probability_array = probability.squeeze().cpu().numpy()

        predicted_mask = (
            probability_array > 0.5
        ).astype(np.uint8)

        ground_truth_mask = gt_mask[0]

        original_image = cv2.imread(
            valid_df.iloc[index]["sat_image_path"]
        )

        if original_image is None:
            print(f"Skipping unreadable validation image {index}.")
            continue

        original_image = cv2.cvtColor(
            original_image,
            cv2.COLOR_BGR2RGB
        )

        original_image = cv2.resize(
            original_image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        visualize(
            original_image=original_image,
            ground_truth_road_mask=ground_truth_mask,
            predicted_road_mask=predicted_mask,
            prediction_probability=probability_array
        )

        prediction_path = sample_preds_folder / f"sample_pred_{index}.png"

        cv2.imwrite(
            str(prediction_path),
            predicted_mask * 255
        )

    print("\nAll predictions generated successfully!")
    print("Model location:", MODEL_PATH)
    print("Prediction folder:", sample_preds_folder)
