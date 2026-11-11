"""Audit saved results, split isolation, model/demo parity, and report completeness."""
import csv
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from app import create_app
from traffic import score_predictions

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    artifacts = HERE / 'artifacts'
    results = artifacts / 'results'
    manifest = json.loads((results / 'manifest.json').read_text())
    for name, expected in manifest['code_hashes'].items():
        require(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected,
                f'Final source hash mismatch: {name}')
    if 'training_code_hashes' in manifest:
        for name, expected in manifest['training_code_hashes'].items():
            source = results / 'training-source' / name
            if not source.exists():
                source = HERE / name
            require(hashlib.sha256(source.read_bytes()).hexdigest() == expected,
                    f'Training source hash mismatch: {name}')
    metrics = json.loads((results / 'metrics.json').read_text())
    training = json.loads((results / 'training.json').read_text())
    predictions = np.load(results / 'predictions.npz')
    truth = predictions['truth']
    require(len(truth) == 12630, 'Expected the entire official 12,630-image test set')
    require(set(truth.tolist()) == set(range(43)), 'Missing test classes')
    require(len(metrics) == 30, 'Expected 3 models x 10 conditions')
    unique = {(r['model'], r['corruption'], r['severity']) for r in metrics}
    require(len(unique) == 30, 'Duplicate evaluation conditions')
    for row in metrics:
        key = f"{row['model']}__{row['corruption']}__{row['severity']}"
        predicted = predictions[key]
        require(predicted.shape == truth.shape, f'Prediction count mismatch: {key}')
        require(((predicted >= 0) & (predicted < 43)).all(), f'Invalid class ID: {key}')
        score = score_predictions(truth, predicted)
        for field in ['accuracy', 'macro_f1']:
            require(abs(score[field] - row[field]) < 1e-10, f'Metric mismatch: {key} {field}')
        require(row['n'] == 12630, f'Incomplete evaluation: {key}')
    with (results / 'split.csv').open(newline='', encoding='utf-8') as stream:
        split = list(csv.DictReader(stream))
    train = [r for r in split if r['partition'] == 'train']
    validation = [r for r in split if r['partition'] == 'validation']
    require(len(train) == manifest['training_images'], 'Training split mismatch')
    require(len(validation) == manifest['validation_images'], 'Validation split mismatch')
    require(len(split) == 39209, 'Training-partition image count mismatch')
    require(len({r['path'] for r in split}) == len(split), 'Duplicate image in split')
    require(not ({r['track_id'] for r in train} & {r['track_id'] for r in validation}), 'Sign track leakage')
    for partition in [train, validation]:
        require({int(r['class_id']) for r in partition} == set(range(43)), 'Split omits a class')
    for name in ['cnn_clean', 'cnn_augmented']:
        history = json.loads((results / f'{name}_history.json').read_text())
        require(len(history) == manifest['epochs'], 'Incomplete CNN training history')
        require(abs(max(r['val_macro_f1'] for r in history) - training[name]['best_validation_macro_f1']) < 1e-10,
                'Best checkpoint metadata mismatch')
    require(training['hog_svm']['converged'], 'Linear SVM did not converge')
    require(len(list((artifacts / 'samples').glob('*.png'))) == 43, 'Incomplete sample gallery')
    client = create_app(artifacts).test_client()
    require(client.get('/').status_code == 200, 'Landing page unavailable')
    require(client.get('/api/health').json['ready'], 'Models unavailable')
    require(client.get('/api/results').status_code == 200, 'Metrics endpoint unavailable')
    for class_id in [0, 14, 42]:
        response = client.post('/api/predict', data={'sample': str(class_id), 'corruption': 'clean', 'severity': '0'})
        require(response.status_code == 200, f'Demo prediction failed for {class_id}')
        index = int(np.flatnonzero(truth == class_id)[0])
        for model in response.json['predictions']:
            require(model['top'][0]['class_id'] == int(predictions[f"{model['model']}__clean__0"][index]),
                    'Demo differs from offline prediction')
    with (artifacts / 'samples' / '14.png').open('rb') as image:
        response = client.post('/api/predict', data={'image': (image, 'stop.png'),
                                                   'corruption': 'noise', 'severity': '3'},
                               content_type='multipart/form-data')
    require(response.status_code == 200 and response.json['true_class'] is None, 'Image upload failed')
    require(len(response.json['predictions']) == 3, 'Missing model predictions')
    reader = PdfReader(HERE / 'report' / 'report.pdf')
    text = '\n'.join(page.extract_text() for page in reader.pages)
    require(4 <= len(reader.pages) <= 7, 'Report length outside course limits')
    require('@@' not in text, 'Unfilled report value')
    compact_text = ''.join(unicodedata.normalize('NFKC', text).upper().split())
    for section in ['INTRODUCTION', 'TECHNICAL APPROACH', 'EXPERIMENTAL SETUP', 'RESULTS AND ANALYSIS', 'CONCLUSION', 'REFERENCES']:
        require(section.replace(' ', '') in compact_text, f'Missing report section: {section}')
    for model in ['hog_svm', 'cnn_clean', 'cnn_augmented']:
        clean = next(r for r in metrics if r['model'] == model and r['corruption'] == 'clean')
        require(f"{clean['accuracy'] * 100:.2f}" in text, 'Report omits actual clean accuracy')
    require(client.get('/report').status_code == 200, 'Report endpoint unavailable')
    print('VERIFIED: all 30 metrics recomputed from full-test predictions; track-disjoint split;')
    print('complete training; model/demo parity; uploads; sample gallery; compiled IEEE report.')
    print(f'Report pages: {len(reader.pages)}. Author details and code URL intentionally omitted.')


if __name__ == '__main__':
    main()
