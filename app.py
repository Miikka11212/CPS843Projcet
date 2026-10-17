"""Local traffic sign robustness demo. Run with python app.py."""
import argparse
import base64
import io
import json
import os
import threading
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, UnidentifiedImageError
import torch

from traffic import CORRUPTIONS, corrupt, hog_features, load_cnn, predict_cnn, prepare_image

HERE = Path(__file__).resolve().parent


def image_data(image):
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def create_app(artifacts=None):
    artifacts = Path(artifacts or HERE / 'artifacts')
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
    inference_lock = threading.Lock()

    @lru_cache(maxsize=1)
    def models():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_num_threads(4)
        return (json.loads((artifacts / 'labels.json').read_text()),
                joblib.load(artifacts / 'models' / 'hog_svm.joblib'),
                {name: load_cnn(artifacts / 'models' / f'{name}.pt', device)
                 for name in ['cnn_clean', 'cnn_augmented']}, device)

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/health')
    def health():
        required = [artifacts / 'labels.json', artifacts / 'models' / 'hog_svm.joblib',
                    *[artifacts / 'models' / f'{name}.pt' for name in ['cnn_clean', 'cnn_augmented']]]
        return jsonify(product='signlab', pid=os.getpid(), ready=all(path.is_file() for path in required), classes=43,
                       corruptions=CORRUPTIONS)

    @app.get('/api/results')
    def results():
        try:
            return jsonify(metrics=json.loads((artifacts / 'results' / 'metrics.json').read_text()),
                           labels=json.loads((artifacts / 'labels.json').read_text()))
        except FileNotFoundError:
            return jsonify(error='Run run_experiment.py to generate results.'), 503

    @app.get('/samples/<int:sample>')
    def sample_file(sample):
        path = artifacts / 'samples' / f'{sample:02d}.png'
        if not 0 <= sample < 43 or not path.is_file():
            return jsonify(error='Sample not found'), 404
        return send_file(path, mimetype='image/png')

    @app.get('/report')
    def report_file():
        path = HERE / 'report' / 'report.pdf'
        if not path.is_file():
            return jsonify(error='Report not generated yet'), 404
        return send_file(path, mimetype='application/pdf')

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error='Image exceeds the 8 MB upload limit.'), 413

    @app.post('/api/predict')
    def predict():
        kind = request.form.get('corruption', 'clean')
        try:
            severity = int(request.form.get('severity', '0'))
            if kind not in ['clean', *CORRUPTIONS] or severity not in range(4):
                raise ValueError('Choose a valid corruption and severity from 0 to 3.')
            if 'image' in request.files:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', Image.DecompressionBombWarning)
                    with Image.open(request.files['image'].stream) as source:
                        if source.width * source.height > 20_000_000:
                            raise ValueError('Image must contain at most 20 million pixels.')
                        original = prepare_image(source)
                truth = None
            else:
                sample = int(request.form.get('sample', '14'))
                if not 0 <= sample < 43:
                    raise ValueError('Sample must be a class ID from 0 to 42.')
                path = artifacts / 'samples' / f'{sample:02d}.png'
                if not path.is_file():
                    return jsonify(error='Sample images are missing. Run the experiment first.'), 503
                with Image.open(path) as source:
                    original = prepare_image(source)
                truth = sample
        except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError,
                Image.DecompressionBombWarning):
            return jsonify(error='Use a valid image (up to 20 MP), sample 0-42, and severity 0-3.'), 400
        changed = corrupt(original, kind, severity)
        try:
            with inference_lock:
                names, svm, networks, device = models()
                margin = svm.decision_function(hog_features(changed[None]))[0]
                output = []
                top = np.argsort(margin)[-3:][::-1]
                output.append({'model': 'hog_svm', 'score_type': 'SVM margin (not probability)',
                               'top': [{'class_id': int(i), 'label': names[i], 'score': float(margin[i])} for i in top]})
                for name, model in networks.items():
                    probabilities = predict_cnn(model, changed[None], device)[0]
                    top = np.argsort(probabilities)[-3:][::-1]
                    output.append({'model': name, 'score_type': 'Softmax probability (uncalibrated)',
                                   'top': [{'class_id': int(i), 'label': names[i], 'score': float(probabilities[i])} for i in top]})
        except FileNotFoundError:
            return jsonify(error='Models are missing. Run run_experiment.py first.'), 503
        return jsonify(predictions=output, original=image_data(original), transformed=image_data(changed),
                       true_class=truth, true_label=names[truth] if truth is not None else None)

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8430)
    parser.add_argument('--artifacts', type=Path, default=HERE / 'artifacts')
    args = parser.parse_args()
    create_app(args.artifacts).run(host='127.0.0.1', port=args.port, debug=False)
