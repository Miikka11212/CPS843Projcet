"""Start a hidden local server, wait for readiness, and open the demo."""
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

HERE = Path(__file__).resolve().parent
URL = 'http://127.0.0.1:8430'


def running():
    try:
        with urllib.request.urlopen(URL + '/api/health', timeout=2) as response:
            data = json.load(response)
            return data.get('ready') is True and data.get('classes') == 43 and data.get('product') == 'signlab'
    except (OSError, urllib.error.URLError, ValueError):
        return False


def main():
    if '--stop' in sys.argv:
        try:
            with urllib.request.urlopen(URL + '/api/health', timeout=2) as response:
                data = json.load(response)
            if data.get('product') != 'signlab':
                raise SystemExit('Port 8430 belongs to another application; nothing was stopped.')
            os.kill(int(data['pid']), signal.SIGTERM)
            print('SignLab stopped.')
        except (OSError, urllib.error.URLError):
            print('SignLab is not running.')
        return
    if not running():
        with (HERE / 'demo.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, str(HERE / 'app.py')], cwd=HERE,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        for _ in range(120):
            if running():
                break
            if process.poll() is not None:
                raise SystemExit('Server stopped. See demo.log for details.')
            time.sleep(.5)
        else:
            process.terminate()
            raise SystemExit('Server did not become ready. Check trained models and demo.log.')
    webbrowser.open(URL)
    print(f'SignLab is running at {URL}. Double-click stop_demo.bat to stop it.')


if __name__ == '__main__':
    main()
