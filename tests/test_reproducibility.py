import unittest


class ProvenanceTests(unittest.TestCase):
    def test_checkpoint_reuse_rejects_changed_runtime(self):
        from run_experiment import check_compatibility
        original = {key: 'same' for key in ['seed', 'image_size', 'epochs', 'batch_size',
                                           'source_hashes', 'code_hashes', 'device', 'torch',
                                           'python', 'gpu', 'platform', 'libraries']}
        self.assertIsNone(check_compatibility(original, dict(original)))
        for key in ['device', 'torch', 'python', 'gpu', 'platform', 'libraries', 'code_hashes']:
            changed = dict(original)
            changed[key] = 'different'
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    check_compatibility(original, changed)
        self.assertEqual(original['device'], 'same')


if __name__ == '__main__':
    unittest.main()
