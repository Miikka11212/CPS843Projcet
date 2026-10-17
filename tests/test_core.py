import importlib.util
import tempfile
import unittest
from pathlib import Path


class CoreTests(unittest.TestCase):
    def test_project_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('traffic'))

    def test_csv_paths_are_portable_and_cannot_escape_root(self):
        from traffic import resolve_image
        root = Path(tempfile.gettempdir()) / 'signs'
        self.assertEqual(resolve_image(root, '/code/Train/20/a.png'), root / 'Train/20/a.png')
        self.assertEqual(resolve_image(root, r'Test\00000.png'), root / 'Test/00000.png')
        with self.assertRaises(ValueError):
            resolve_image(root, '../../outside.png')

    def test_validation_has_no_shared_sign_tracks(self):
        import numpy as np
        from traffic import split_tracks, track_id
        paths = [f'Train/{c}/{c:05d}_{t:05d}_{f:05d}.png'
                 for c in range(3) for t in range(10) for f in range(3)]
        labels = np.repeat(np.arange(3), 30)
        train, val = split_tracks(paths, labels, seed=42)
        self.assertFalse(set(track_id(paths[i]) for i in train) & set(track_id(paths[i]) for i in val))
        self.assertEqual(set(labels[train]), {0, 1, 2})
        self.assertEqual(set(labels[val]), {0, 1, 2})
        self.assertEqual(len(train) + len(val), len(paths))
        self.assertTrue(np.array_equal(val, split_tracks(paths, labels, seed=42)[1]))

    def test_corruptions_preserve_clean_and_are_deterministic(self):
        import numpy as np
        from traffic import corrupt
        image = np.full((32, 32, 3), 120, dtype=np.uint8)
        for kind in ['clean', 'blur', 'darkness', 'noise']:
            self.assertTrue(np.array_equal(corrupt(image, kind, 0), image))
        self.assertLess(corrupt(image, 'darkness', 3).mean(), image.mean())
        self.assertTrue(np.array_equal(corrupt(image, 'noise', 3, 10), corrupt(image, 'noise', 3, 10)))
        self.assertEqual(corrupt(image, 'noise', 3).dtype, np.uint8)
        with self.assertRaises(ValueError):
            corrupt(image, 'unknown', 1)
        with self.assertRaises(ValueError):
            corrupt(image, 'noise', 4)

    def test_cnn_and_hog_support_batches(self):
        import numpy as np
        import torch
        from traffic import SignCNN, hog_features, predict_cnn
        images = np.full((3, 32, 32, 3), 120, dtype=np.uint8)
        features = hog_features(images)
        self.assertEqual(features.shape, (3, 324))
        self.assertTrue(np.isfinite(features).all())
        model = SignCNN()
        self.assertEqual(tuple(model(torch.zeros(3, 3, 32, 32)).shape), (3, 43))
        probabilities = predict_cnn(model, images, 'cpu')
        self.assertEqual(probabilities.shape, (3, 43))
        self.assertTrue(np.allclose(probabilities.sum(1), 1, atol=1e-6))

    def test_metrics_retain_all_43_classes(self):
        from traffic import score_predictions
        score = score_predictions([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(score['accuracy'], .75)
        self.assertEqual(len(score['confusion_matrix']), 43)
        self.assertGreater(score['macro_f1'], 0)


if __name__ == '__main__':
    unittest.main()
