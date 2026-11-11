#!/bin/sh
set -eu
cd "$(dirname "$0")"

export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/ and reopen this launcher." >&2
    exit 1
fi

if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -r requirements.txt
exec .venv/bin/python start_demo.py "$@"
