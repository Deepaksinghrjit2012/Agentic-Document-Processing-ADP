"""Isolated local inference with a structured result for the upload service."""
import json
import sys
from pathlib import Path
from adp import run

if __name__ == '__main__':
    document, schema, classification, device, output = sys.argv[1:6]
    mode = sys.argv[6] if len(sys.argv)>6 else 'extract'
    try:
        if mode=='classify':
            from classification import run as classify
            result=classify(document,json.loads(Path(schema).read_text()),device)
        else:
            result = run(document, json.loads(Path(schema).read_text()), device,
                         json.loads(Path(classification).read_text()),
                         ocr_pages=json.loads(Path(sys.argv[7]).read_text()) if len(sys.argv)>7 else None)
        envelope = {'result': result}
    except Exception as error:
        from progress import report
        report(f'Worker failed: {type(error).__name__}')
        envelope = {'error': str(error)}
    Path(output).write_text(json.dumps(envelope))
