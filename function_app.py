"""Azure Functions v2 ASGI adapter; use a function key for hosted requests."""
import os
import tempfile
from pathlib import Path

# Deployment packages are read-only. Workers inherit these writable directories.
runtime = Path(tempfile.gettempdir()) / 'adp'
os.environ.setdefault('ADP_LOG_DIR', str(runtime / 'logs'))
os.environ.setdefault('ADP_OUTPUT_DIR', str(runtime / 'outputs'))
# Leave time to return an error before Azure's 230-second HTTP timeout.
os.environ.setdefault('ADP_REQUEST_TIMEOUT', '210')

import azure.functions as func
from server import app as api

app = func.AsgiFunctionApp(app=api, http_auth_level=func.AuthLevel.FUNCTION)
