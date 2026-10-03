"""Bounded progress feed and rotating local log, without extracted document text."""
import logging
import os
import threading
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

events = deque(maxlen=500)
event_lock = threading.Lock()
logger = logging.getLogger('adp.progress')
logger.setLevel(logging.INFO)
logger.propagate = bool(os.environ.get('FUNCTIONS_WORKER_RUNTIME'))
if not logger.handlers:
    folder = Path(os.environ.get('ADP_LOG_DIR', str(Path(__file__).parent / 'logs')))
    folder.mkdir(parents=True,exist_ok=True)
    handler = RotatingFileHandler(folder / 'adp.log', maxBytes=2_000_000, backupCount=2)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())

def report(message):
    event = {'time': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(timespec='seconds'), 'message': message}
    with event_lock:
        events.append(event)
    logger.info('%s %s', event['time'], message)

def snapshot():
    # Inference workers are separate processes; read the shared stage log.
    path = Path(os.environ.get('ADP_LOG_DIR', str(Path(__file__).parent / 'logs'))) / 'adp.log'
    if not path.exists():
        return []
    result = []
    for line in path.read_text(errors='replace').splitlines()[-500:]:
        timestamp, separator, message = line.partition(' ')
        if separator:
            result.append({'time': timestamp, 'message': message})
    return result
