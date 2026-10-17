import importlib.util
import io
import tempfile
import unittest
from pathlib import Path


class DemoTests(unittest.TestCase):
    def test_app_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('app'))

    def test_landing_page_and_validation_without_models(self):
        from app import create_app
        with tempfile.TemporaryDirectory() as directory:
            client = create_app(Path(directory)).test_client()
            self.assertEqual(client.get('/').status_code, 200)
            self.assertEqual(client.get('/api/health').json['ready'], False)
            for values in [{'severity': '9'}, {'severity': 'bad'}, {'corruption': 'other'}, {'sample': '../secret'}]:
                response = client.post('/api/predict', data=values)
                self.assertEqual(response.status_code, 400)
                self.assertIn('error', response.json)
            response = client.post('/api/predict', data={
                'image': (io.BytesIO(b'not an image'), 'fake.png')}, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 400)
            self.assertEqual(client.get('/api/results').status_code, 503)
            self.assertEqual(client.get('/samples/99').status_code, 404)


if __name__ == '__main__':
    unittest.main()
