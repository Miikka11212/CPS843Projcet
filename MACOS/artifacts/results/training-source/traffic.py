"""Shared data handling, image transformations, features, and CNN inference."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from skimage.feature import hog
import torch
from torch import nn

SIZE = 32
CLASSES = 43
SEED = 42
CORRUPTIONS = {
    'blur': [0.6, 1.2, 1.8],
    'darkness': [0.65, 0.40, 0.20],
    'noise': [0.04, 0.08, 0.12],
}


def resolve_image(root, raw_path):
    """Accept Kaggle /code/ paths, but never allow paths outside the dataset."""
    root = Path(root).resolve()
    relative = str(raw_path).replace('\\', '/')
    if relative.startswith('/code/'):
        relative = relative[len('/code/'):]
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f'Image path escapes dataset: {raw_path}')
    return candidate


def read_rows(root, split):
    with (Path(root) / f'{split}.csv').open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {'Path', 'ClassId'}.issubset(rows[0]):
        raise ValueError(f'{split}.csv must contain Path and ClassId columns')
    for row in rows:
        if not 0 <= int(row['ClassId']) < CLASSES:
            raise ValueError('ClassId must be between 0 and 42')
    return rows


def read_sign_names(root):
    with (Path(root) / 'signnames.csv').open(newline='', encoding='utf-8-sig') as stream:
        names = {int(r['ClassId']): r['SignName'] for r in csv.DictReader(stream)}
    if set(names) != set(range(CLASSES)):
        raise ValueError('signnames.csv must label all 43 classes')
    return [names[i] for i in range(CLASSES)]


def track_id(path):
    # GTSRB names encode class_track_frame; frames of one sign stay together.
    pieces = Path(str(path).replace('\\', '/')).stem.split('_')
    if len(pieces) != 3 or not all(p.isdigit() for p in pieces):
        raise ValueError(f'Expected class_track_frame image name: {path}')
    return '_'.join(pieces[:2])


def split_tracks(paths, labels, seed=SEED, validation_fraction=.2):
    labels = np.asarray(labels)
    if len(paths) != len(labels) or not 0 < validation_fraction < 1:
        raise ValueError('Invalid split inputs')
    groups = np.array([track_id(p) for p in paths])
    random = np.random.default_rng(seed)
    validation_groups = set()
    for class_id in np.unique(labels):
        unique = np.unique(groups[labels == class_id])
        if len(unique) < 2:
            raise ValueError(f'Class {class_id} needs at least two independent tracks')
        count = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
        validation_groups.update(random.permutation(unique)[:count].tolist())
    mask = np.array([g in validation_groups for g in groups])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def prepare_image(image):
    """Full sign crop resized with bilinear interpolation; no ROI recropping."""
    return np.asarray(ImageOps.exif_transpose(image).convert('RGB').resize(
        (SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.uint8).copy()


def load_images(root, rows):
    result = np.empty((len(rows), SIZE, SIZE, 3), dtype=np.uint8)
    for index, row in enumerate(rows):
        with Image.open(resolve_image(root, row['Path'])) as image:
            result[index] = prepare_image(image)
    return result


def corrupt(image, kind='clean', severity=0, seed=SEED):
    if kind not in ['clean', *CORRUPTIONS] or severity not in range(4):
        raise ValueError('Unknown corruption or severity; severity must be 0..3')
    image = np.asarray(image, dtype=np.uint8)
    if kind == 'clean' or severity == 0:
        return image.copy()
    value = CORRUPTIONS[kind][severity - 1]
    if kind == 'blur':
        return np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(value))).copy()
    if kind == 'darkness':
        return np.clip(image.astype(np.float32) * value, 0, 255).astype(np.uint8)
    random = np.random.default_rng(seed)
    return np.clip(image.astype(np.float32) + random.normal(0, value * 255, image.shape),
                   0, 255).astype(np.uint8)


def corrupt_batch(images, kind, severity, seed=SEED):
    return np.stack([corrupt(im, kind, severity, seed + i) for i, im in enumerate(images)])


def hog_features(images):
    # One HOG descriptor = 3*3 blocks * 2*2 cells * 9 orientations = 324.
    output = np.empty((len(images), 324), dtype=np.float32)
    for index, image in enumerate(images):
        gray = np.asarray(Image.fromarray(image).convert('L'), dtype=np.float32) / 255
        output[index] = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), block_norm='L2-Hys')
    return output


class SignCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 3
        for width in (32, 64, 128):
            layers.extend([nn.Conv2d(channels, width, 3, padding=1), nn.BatchNorm2d(width),
                           nn.ReLU(), nn.Conv2d(width, width, 3, padding=1),
                           nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)])
            channels = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256),
                                        nn.ReLU(), nn.Dropout(.3), nn.Linear(256, CLASSES))

    def forward(self, images):
        return self.classifier(self.features(images))


def image_tensor(images, device='cpu'):
    return torch.as_tensor(np.ascontiguousarray(images), device=device).permute(
        0, 3, 1, 2).float().div_(255)


@torch.inference_mode()
def predict_cnn(model, images, device='cpu', batch_size=512):
    model.eval()
    results = []
    for start in range(0, len(images), batch_size):
        tensor = image_tensor(images[start:start + batch_size], device)
        results.append(model(tensor).softmax(1).cpu().numpy())
    return np.concatenate(results)


def load_cnn(path, device='cpu'):
    model = SignCNN().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


def score_predictions(truth, predictions):
    return {
        'accuracy': float(accuracy_score(truth, predictions)),
        'macro_f1': float(f1_score(truth, predictions, labels=range(CLASSES),
                                 average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(truth, predictions, labels=range(CLASSES)).tolist(),
    }


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
