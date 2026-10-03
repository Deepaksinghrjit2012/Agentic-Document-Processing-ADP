"""Loopback-only upload UI for local ADP. Serialize requests to bound memory."""
import json
import os
import tempfile
import threading
import subprocess
import sys
import time
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from evaluation import extraction_payload
from progress import report, snapshot

from typing import Any
from pydantic import BaseModel

class ExtractionResponse(BaseModel):
    extracted_fields: dict[str, Any]

class ClassificationResult(BaseModel):
    document_type: str
    confidence: float
    probabilities: dict[str,float]

class ClassificationResponse(BaseModel):
    classification: ClassificationResult
    ocr_text: str
    ocr_pages: list[dict[str, Any]]

app = FastAPI(title='ADP Extraction and Classification API', version='1.0.0')
lock = threading.Lock()
cancel_requested = threading.Event()

def run_worker(path, schema, classification, mode="extract", ocr_pages=None):
    folder = path.parent
    for name, value in [('schema', schema), ('classification', classification)]:
        (folder / (name+'.json')).write_text(json.dumps(value))
    (folder/'ocr.json').write_text(json.dumps(ocr_pages))
    output = folder / 'result.json'
    root = Path(__file__).parent
    worker_env = os.environ.copy()
    worker_env['PYTHONPATH'] = os.pathsep.join(str(entry) for entry in sys.path if entry)
    proc = subprocess.Popen([sys.executable, str(root/'worker.py'), str(path),
        str(folder/'schema.json'), str(folder/'classification.json'),
        os.environ.get('ADP_DEVICE','cpu'), str(output), mode, str(folder/'ocr.json')], cwd=root, env=worker_env)
    start = time.monotonic()
    next_heartbeat = start + 15
    timeout = float(os.environ.get('ADP_REQUEST_TIMEOUT','900'))
    try:
        while True:
            if cancel_requested.is_set():
                raise RuntimeError('Extraction canceled')
            if time.monotonic()-start >= timeout:
                raise RuntimeError('Extraction exceeded request timeout; worker stopped')
            now = time.monotonic()
            if now >= next_heartbeat:
                report(f'Inference worker alive; elapsed {int(now-start)} seconds')
                next_heartbeat = now + 15
            try:
                proc.wait(timeout=2)
                break
            except subprocess.TimeoutExpired:
                continue
        if not output.exists():
            raise RuntimeError('Inference worker exited without a result; inspect server.log')
        envelope = json.loads(output.read_text())
        if 'error' in envelope:
            raise RuntimeError(envelope['error'])
        return envelope['result']
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

@app.get('/', response_class=HTMLResponse)
def home():
    schema = (Path(__file__).parent / 'examples/invoice.schema.json').read_text()
    import html
    return '''<!doctype html><html><head><title>Local ADP</title><style>
    body{font:16px system-ui;max-width:850px;margin:45px auto;padding:20px;background:#f6f7fb;color:#172235}
    textarea{width:100%;height:260px}button{padding:12px;margin:16px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}
    </style></head><body><h1>Agentic Document Processing</h1>
    <p>Extract document fields with GPT and Jev support scores.</p>
    <form id="upload"><p><input name="file" type="file" accept="application/pdf,image/*" required></p>
    <label>JSON Schema<textarea name="schema">''' + html.escape(schema) + '''</textarea></label>
    <button id="submit">Extract fields</button></form>
    <button type="button" id="cancel" onclick="fetch('/cancel',{method:'POST'})">Cancel current extraction</button>
    <h2>Live progress</h2><p id="status"></p><pre id="logs" style="max-height:260px;overflow:auto"></pre>
    <h2>Result</h2><pre id="result"></pre>
    <p id="review"></p>
    <script>async function refreshLogs(){try{const res=await fetch('/logs');const data=await res.json();
    const l=document.querySelector('#logs');l.textContent=data.events.map(e=>e.time+' '+e.message).join('\\n');
    document.querySelector('#status').textContent=data.running?'Processing…':'Idle';l.scrollTop=l.scrollHeight;
    }catch(err){document.querySelector('#status').textContent='Progress connection unavailable';}}
    setInterval(refreshLogs,1000);refreshLogs();</script>
    <script>document.querySelector('form').onsubmit=async e=>{e.preventDefault();
    const b=document.querySelector('button'),r=document.querySelector('#result');b.disabled=true;
    r.textContent='Processing. This can take several minutes.';
    document.querySelector('#review').textContent='';
    try{const res=await fetch('/extract',{method:'POST',body:new FormData(e.target)});
    r.textContent=JSON.stringify(await res.json(),null,2);
    if(res.ok){
    document.querySelector('#review').textContent=res.headers.get('X-Needs-Review')==='true'?'Extraction needs review.':'';
    }}catch(err){r.textContent=String(err);}
    finally{b.disabled=false;}};</script></body></html>'''

def process_upload(file, schema, mode='extract', ocr_pages=None):
    if not lock.acquire(blocking=False):
        raise HTTPException(429, 'Another document is processing; try again when it completes')
    try:
        cancel_requested.clear()
        report('Upload received; validating document and schema')
        content = file.file.read(30 * 1024 * 1024 + 1)
        if len(content) > 30 * 1024 * 1024:
            raise HTTPException(413, 'Maximum file size is 30MB')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'document'
            path.write_bytes(content)
            if mode=='classify':
                result=run_worker(path,json.loads(schema),{},mode='classify')
                payload={key:result[key] for key in ('classification','ocr_text','ocr_pages')}
            else:
                if ocr_pages is not None:
                    supplied=json.loads(ocr_pages)
                    if isinstance(supplied,dict):
                        supplied=supplied.get('ocr_pages')
                    if not isinstance(supplied,list):
                        raise ValueError('Supply an OCR page array or a classification response containing ocr_pages')
                    result = run_worker(path,json.loads(schema),{},ocr_pages=supplied)
                else:
                    result = run_worker(path, json.loads(schema), {})
                payload=extraction_payload(result)
            return JSONResponse(payload, headers={
                'X-Needs-Review': str(result['needs_review']).lower()})
    except (ValueError, RuntimeError) as error:
        report(f'Failed: {type(error).__name__}; see extraction result for details')
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        report(f'Failed: {type(error).__name__}; check server terminal for details')
        raise
    finally:
        file.file.close()
        lock.release()

@app.post('/extract', response_model=ExtractionResponse, summary='Extract fields with Jev confidence')
@app.post('/api/analyze', response_model=ExtractionResponse)
def extract(file: UploadFile = File(...), schema: str = Form(...), ocr_pages: str | None = Form(None, description="Paste the entire /classify JSON response or its ocr_pages array. Initial OCR calls are skipped when provided.")):
    return process_upload(file,schema,ocr_pages=ocr_pages)

@app.post('/classify', response_model=ClassificationResponse, summary='Classify document using caller-supplied Jev questions')
@app.post('/api/classify', response_model=ClassificationResponse)
def classify(file: UploadFile = File(...), schema: str = Form(...)):
    return process_upload(file,schema,mode='classify')

@app.get('/logs')
def logs():
    return {'running': lock.locked(), 'events': snapshot()}

@app.post('/cancel')
def cancel():
    if lock.locked():
        cancel_requested.set()
        report('Cancellation requested; stopping inference worker')
        return {'cancel_requested': True}
    return {'cancel_requested': False}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000, access_log=False)
