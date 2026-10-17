"""Train, evaluate, and export the complete GTSRB robustness experiment."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import torch
from torch import nn
from torch.nn import functional as F

from traffic import (CLASSES, CORRUPTIONS, SEED, SIZE, SignCNN, corrupt_batch,
                     hog_features, image_tensor, load_cnn, load_images, predict_cnn,
                     read_rows, read_sign_names, save_json, score_predictions,
                     split_tracks, track_id)


def augment(images):
    """Randomly apply one of clean, blur, darkness, or noise, on the GPU."""
    images = images.clone()
    choices = torch.randint(4, (len(images),), device=images.device)
    dark = choices == 2
    images[dark] *= .2 + .7 * torch.rand((int(dark.sum()), 1, 1, 1), device=images.device)
    noisy = choices == 3
    noise_images = images[noisy]
    sigmas = .02 + .12 * torch.rand((len(noise_images), 1, 1, 1), device=images.device)
    images[noisy] = noise_images + torch.randn_like(noise_images) * sigmas
    blurred = choices == 1
    if blurred.any():
        sigma = float(torch.empty(1).uniform_(.4, 1.8))
        coordinates = torch.arange(-6, 7, device=images.device)
        kernel = torch.exp(-coordinates.square() / (2 * sigma ** 2))
        kernel /= kernel.sum()
        horizontal = kernel.reshape(1, 1, 1, 13).expand(3, 1, 1, 13)
        vertical = kernel.reshape(1, 1, 13, 1).expand(3, 1, 13, 1)
        selected = F.pad(images[blurred], (6, 6, 6, 6), mode='replicate')
        images[blurred] = F.conv2d(F.conv2d(selected, horizontal, groups=3), vertical, groups=3)
    return images.clamp_(0, 1)


def train_cnn(images, labels, train_ids, val_ids, augmented, args, destination):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = SignCNN().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.00001)
    loss_function = nn.CrossEntropyLoss()
    # All 32x32 training images occupy <0.5 GB as floats; avoid Windows worker overhead.
    train_images = image_tensor(images[train_ids], args.device)
    train_labels = torch.as_tensor(labels[train_ids], device=args.device)
    best_f1 = -1.0
    history = []
    start_time = time.perf_counter()
    name = 'cnn_augmented' if augmented else 'cnn_clean'
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        permutation = torch.randperm(len(train_ids), device=args.device)
        total_loss, correct = 0., 0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start:start + args.batch_size]
            inputs, targets = train_images[indices], train_labels[indices]
            if augmented:
                inputs = augment(inputs)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            correct += int((logits.argmax(1) == targets).sum())
        scheduler.step()
        validation = predict_cnn(model, images[val_ids], args.device).argmax(1)
        metrics = score_predictions(labels[val_ids], validation)
        record = {'epoch': epoch, 'train_loss': total_loss / len(train_ids),
                  'train_accuracy': correct / len(train_ids),
                  'val_accuracy': metrics['accuracy'], 'val_macro_f1': metrics['macro_f1'],
                  'learning_rate': optimizer.param_groups[0]['lr'],
                  'seconds': time.perf_counter() - epoch_start}
        history.append(record)
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                        'val_macro_f1': best_f1, 'augmented': augmented,
                        'classes': CLASSES, 'image_size': SIZE}, destination)
        print(f"{name} epoch {epoch:02d}/{args.epochs}: "
              f"loss={record['train_loss']:.4f} val_acc={metrics['accuracy']:.4f} "
              f"val_f1={best_f1:.4f} ({record['seconds']:.1f}s)", flush=True)
        save_json(args.output / 'results' / f'{name}_history.json', history)
    elapsed = time.perf_counter() - start_time
    return {'training_seconds': elapsed, 'parameters': sum(p.numel() for p in model.parameters()),
            'best_validation_macro_f1': best_f1, 'epochs': args.epochs,
            'best_epoch': torch.load(destination, map_location='cpu', weights_only=True)['epoch']}


def write_predictions(path, rows, truth, predictions):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'true_class', 'predicted_class'])
        writer.writerows((r['Path'], int(y), int(p)) for r, y, p in zip(rows, truth, predictions))


def check_compatibility(previous, current):
    # Reused weights must retain the runtime under which their training was recorded.
    for key in ['seed', 'image_size', 'epochs', 'batch_size', 'source_hashes', 'code_hashes',
                'device', 'torch', 'python', 'gpu', 'platform', 'libraries']:
        if previous[key] != current[key]:
            raise ValueError(f'Existing artifacts have different {key}; use --force or a new --output')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('D:/download/data'))
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--force', action='store_true', help='Retrain models and replace generated results')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in ['models', 'results', 'figures', 'cache', 'samples']:
        (args.output / folder).mkdir(exist_ok=True)
    metadata = {
        'seed': SEED, 'image_size': SIZE, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'device': args.device, 'torch': torch.__version__, 'python': platform.python_version(),
        'platform': platform.platform(), 'dataset_root': str(args.data.resolve()),
        'gpu': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
        'source_hashes': {name: hashlib.sha256((args.data / name).read_bytes()).hexdigest()
                          for name in ['Train.csv', 'Test.csv', 'signnames.csv']},
        'code_hashes': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                        for name in ['traffic.py', 'run_experiment.py']},
        'libraries': {name: importlib.metadata.version(name) for name in
                      ['numpy', 'scikit-learn', 'scikit-image', 'Pillow', 'torch']},
        'corruptions': CORRUPTIONS,
    }
    metadata['training_code_hashes'] = dict(metadata['code_hashes'])
    config_path = args.output / 'results' / 'manifest.json'
    if config_path.exists() and not args.force:
        previous = json.loads(config_path.read_text())
        check_compatibility(previous, metadata)
        metadata['training_code_hashes'] = previous.get('training_code_hashes', previous['code_hashes'])
        if 'training_source_note' in previous:
            metadata['training_source_note'] = previous['training_source_note']
    if args.force or not config_path.exists():
        snapshot = args.output / 'results' / 'training-source'
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in metadata['code_hashes']:
            shutil.copy2(name, snapshot / name)
    print('Loading dataset...', flush=True)
    train_rows, test_rows = read_rows(args.data, 'Train'), read_rows(args.data, 'Test')
    names = read_sign_names(args.data)
    labels = np.array([int(r['ClassId']) for r in train_rows], dtype=np.int64)
    truth = np.array([int(r['ClassId']) for r in test_rows], dtype=np.int64)
    train_ids, val_ids = split_tracks([r['Path'] for r in train_rows], labels)
    metadata.update({'training_images': len(train_ids), 'validation_images': len(val_ids),
                     'test_images': len(truth), 'classes': CLASSES,
                     'training_tracks': len({track_id(train_rows[i]['Path']) for i in train_ids}),
                     'validation_tracks': len({track_id(train_rows[i]['Path']) for i in val_ids})})
    save_json(config_path, metadata)
    save_json(args.output / 'labels.json', names)
    with (args.output / 'results' / 'split.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'class_id', 'track_id', 'partition'])
        val_set = set(val_ids.tolist())
        writer.writerows((r['Path'], int(labels[i]), track_id(r['Path']),
                          'validation' if i in val_set else 'train') for i, r in enumerate(train_rows))
    cache_path = args.output / 'cache' / 'images.npz'
    if cache_path.exists() and not args.force:
        cache = np.load(cache_path)
        images, test_images = cache['train'], cache['test']
    else:
        images, test_images = load_images(args.data, train_rows), load_images(args.data, test_rows)
        np.savez_compressed(cache_path, train=images, test=test_images)
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(truth)} test; device={args.device}", flush=True)
    model_info = {}
    svm_path = args.output / 'models' / 'hog_svm.joblib'
    model_metadata_path = args.output / 'results' / 'training.json'
    if model_metadata_path.exists() and not args.force:
        model_info = json.loads(model_metadata_path.read_text())
    if args.force or not svm_path.exists():
        start = time.perf_counter()
        print('Extracting HOG features and fitting linear SVM...', flush=True)
        features = hog_features(images)
        svm = make_pipeline(StandardScaler(), LinearSVC(C=1., dual='auto', max_iter=10000, random_state=SEED))
        svm.fit(features[train_ids], labels[train_ids])
        val_metrics = score_predictions(labels[val_ids], svm.predict(features[val_ids]))
        joblib.dump(svm, svm_path)
        model_info['hog_svm'] = {'training_seconds': time.perf_counter() - start,
                                  'validation_accuracy': val_metrics['accuracy'],
                                  'best_validation_macro_f1': val_metrics['macro_f1'],
                                  'features': features.shape[1], 'C': 1.0,
                                  'converged': int(np.max(svm[-1].n_iter_)) < 10000}
        save_json(model_metadata_path, model_info)
    for name, augmented in [('cnn_clean', False), ('cnn_augmented', True)]:
        destination = args.output / 'models' / f'{name}.pt'
        if args.force or not destination.exists() or name not in model_info:
            model_info[name] = train_cnn(images, labels, train_ids, val_ids, augmented, args, destination)
            save_json(model_metadata_path, model_info)
    svm = joblib.load(svm_path)
    models = {name: load_cnn(args.output / 'models' / f'{name}.pt', args.device)
              for name in ['cnn_clean', 'cnn_augmented']}
    scores = []
    prediction_arrays = {'truth': truth}
    conditions = [('clean', 0)] + [(kind, level) for kind in CORRUPTIONS for level in (1, 2, 3)]
    for kind, level in conditions:
        changed = test_images if kind == 'clean' else corrupt_batch(test_images, kind, level)
        start = time.perf_counter()
        features = hog_features(changed)
        predictions = {'hog_svm': svm.predict(features)}
        timings = {'hog_svm': time.perf_counter() - start}
        for name, model in models.items():
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            start = time.perf_counter()
            probabilities = predict_cnn(model, changed, args.device)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            timings[name] = time.perf_counter() - start
            predictions[name] = probabilities.argmax(1)
            if kind == 'clean':
                np.save(args.output / 'results' / f'{name}_clean_probabilities.npy', probabilities)
        for name, predicted in predictions.items():
            metrics = score_predictions(truth, predicted)
            record = {'model': name, 'corruption': kind, 'severity': level,
                      'n': len(truth), 'accuracy': metrics['accuracy'],
                      'macro_f1': metrics['macro_f1'], 'inference_seconds': timings[name]}
            scores.append(record)
            prediction_arrays[f'{name}__{kind}__{level}'] = predicted
            if kind == 'clean':
                save_json(args.output / 'results' / f'{name}_confusion.json', metrics['confusion_matrix'])
                write_predictions(args.output / 'results' / f'{name}_clean_predictions.csv', test_rows, truth, predicted)
            print(f"{kind:8s} {level} {name:14s} acc={metrics['accuracy']:.4f} f1={metrics['macro_f1']:.4f}", flush=True)
        save_json(args.output / 'results' / 'metrics.json', scores)
        np.savez_compressed(args.output / 'results' / 'predictions.npz', **prediction_arrays)
    with (args.output / 'results' / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    # A balanced gallery of real official-test images, one sample per class.
    for class_id in range(CLASSES):
        index = int(np.flatnonzero(truth == class_id)[0])
        from PIL import Image
        Image.fromarray(test_images[index]).save(args.output / 'samples' / f'{class_id:02d}.png')
    print('Experiment complete. Run: python make_figures.py', flush=True)


if __name__ == '__main__':
    main()
