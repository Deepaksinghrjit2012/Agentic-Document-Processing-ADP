"""Azure OpenAI Responses adapter for text and document images."""
import base64
import io
import os
import time
import uuid
import threading
from pathlib import Path
from urllib.parse import urlparse
import httpx
from progress import report


def config():
    values = {}
    path = Path(__file__).parent / '.env'
    if path.exists():
        for line in path.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                name, value = line.split('=', 1)
                values[name.strip()] = value.strip().strip('\"\'')
    return lambda name, default='': os.environ.get(name, values.get(name, default))


class AzureModel:
    def __init__(self, role='reasoner', device=None):
        setting = config()
        self.endpoint = setting('AZURE_OPENAI_RESPONSES_ENDPOINT')
        self.deployment = setting('AZURE_OPENAI_DEPLOYMENT', 'gpt-5-mini-1')
        key = setting('AZURE_OPENAI_API_KEY')
        if not self.endpoint or not key:
            raise ValueError('Set AZURE_OPENAI_RESPONSES_ENDPOINT and AZURE_OPENAI_API_KEY in .env')
        if urlparse(self.endpoint).scheme != 'https':
            raise ValueError('Azure OpenAI endpoint must use HTTPS')
        self.request_slots = threading.BoundedSemaphore(int(setting('ADP_MAX_PARALLEL_REQUESTS','16')))
        self.client = httpx.Client(headers={'api-key': key},
            timeout=httpx.Timeout(float(setting('ADP_GENERATION_TIMEOUT', '60')), connect=10, pool=10), follow_redirects=False)

    def generate(self, prompt, images=(), max_tokens=1024, output_schema=None):
        content = [{'type':'input_text', 'text':prompt}]
        for image in images:
            buffer = io.BytesIO()
            image.convert('RGB').save(buffer, format='PNG')
            content.append({'type':'input_image', 'detail':'high',
                'image_url':'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')})
        body = {'model':self.deployment, 'input':[{'role':'user', 'content':content}],
                'reasoning':{'effort':'minimal'}, 'max_output_tokens':max_tokens + 2048, 'store':False}
        if output_schema is not None:
            body['text'] = {'format':{'type':'json_schema','name':'ocr_page',
                                      'strict':True,'schema':output_schema}}
        request_id = uuid.uuid4().hex[:8]
        started = time.monotonic()
        report(f'Azure request {request_id}: started with {len(images)} images')
        try:
            with self.request_slots:
                for attempt in range(3):
                    response = self.client.post(self.endpoint, json=body)
                    if response.status_code not in (429,503) or attempt==2:
                        break
                    try:
                        delay=min(10,max(0,float(response.headers.get('Retry-After',2**attempt))))
                    except ValueError:
                        delay=2**attempt
                    report(f'Azure request {request_id}: throttled or temporarily unavailable; retrying in {delay}s')
                    time.sleep(delay)
        except httpx.TimeoutException:
            report(f'Azure request {request_id}: timed out after {time.monotonic()-started:.1f}s')
            raise RuntimeError('Azure model request timed out') from None
        if response.status_code != 200:
            raise RuntimeError(f'Azure model failed (HTTP {response.status_code}); check key, deployment and endpoint')
        result = response.json()
        if result.get('status') != 'completed':
            raise RuntimeError('Azure model returned an incomplete response; output discarded')
        text = ''.join(part.get('text','') for item in result.get('output', [])
                       if item.get('type') == 'message' for part in item.get('content', [])
                       if part.get('type') == 'output_text')
        if not text.strip():
            raise RuntimeError('Azure model returned no text (possibly refused)')
        report(f'Azure request {request_id}: completed in {time.monotonic()-started:.1f}s')
        return text

    def close(self):
        self.client.close()
