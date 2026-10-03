import json
import pytest
from PIL import Image
import adp

SCHEMA = {'type': 'object', 'properties': {'id': {'type': 'string'}}, 'required': ['id']}

def test_reject_remote_schema_reference():
    with pytest.raises(ValueError, match='local schema'):
        adp.validate_schema({'type': 'object', '$ref': 'https://example.com/schema'})

def test_disagreement_stays_flagged_after_judge_resolution(monkeypatch, tmp_path):
    image = tmp_path / 'document.png'
    Image.new('RGB', (50, 50)).save(image)
    def envelope(value):
        return {'data': {'id': value}, 'evidence': {'id': {'page': 1, 'quote': value}}}
    monkeypatch.setattr(adp, 'gpt_analyze', lambda *args: ([{'text':'ID 001'}], [['ID 001']]))
    responses = iter([ json.dumps(envelope('001')), json.dumps(envelope('01')),
                      json.dumps({**envelope('001'), 'review_fields': []}),
                      'Is extracted_fields.id supported by ocr_text?'])
    class FakeModel:
        def __init__(self, *args): pass
        def generate(self, *args, **kwargs): return next(responses)
        def close(self): pass
    monkeypatch.setattr(adp, 'LocalModel', FakeModel)
    result = adp.run(image, SCHEMA)
    assert result['ocr_text'] == 'ID 001'
    assert result['llm_ocr_text'] == 'ID 001'
    assert result['extracted_fields'] == {'id':'001'}
    assert result['data']['id'] == '001'
    assert result['needs_review']
    assert result['disagreements'] == ['id']

def test_untrusted_json_rejected():
    with pytest.raises(json.JSONDecodeError):
        adp.parse_json('Here is the answer: {"id": "123"}')

def test_malformed_envelope_retry_is_bounded():
    class BadModel:
        calls = 0
        def generate(self, *args):
            self.calls += 1
            return '{"id": "001"}'
    model = BadModel()
    with pytest.raises(ValueError, match='after retry'):
        adp.extract_candidate(model, 'extract')
    assert model.calls == 2


def test_extraction_passes_overlap():
    import threading
    barrier=threading.Barrier(2)
    class Model:
        def generate(self,prompt,images=(),max_tokens=2048):
            if 'reconcile these two' not in prompt:
                barrier.wait(timeout=2)
            return json.dumps({'data':{'id':'001'},'evidence':{'id':{'page':1,'quote':'001'}}})
    task={'start':1,'end':1,'guidance':''}
    result=adp.reconcile(Model(),SCHEMA,[Image.new('RGB',(50,50))],'ID 001',task,[['ID 001']])
    assert result['data']=={'id':'001'}
    assert not result['needs_review']
    assert result['timings']['parallel_extraction_seconds']>=0




def test_supplied_ocr_skips_initial_transcription_but_runs_extraction(monkeypatch,tmp_path):
    image=tmp_path/'document.png'
    Image.new('RGB',(50,50)).save(image)
    def no_ocr(*args):
        raise AssertionError('Initial OCR must be skipped')
    monkeypatch.setattr(adp,'gpt_analyze',no_ocr)
    calls=[]
    class Model:
        def __init__(self,*args): pass
        def generate(self,prompt,images=(),max_tokens=2048,**kwargs):
            calls.append((prompt,len(images)))
            return json.dumps({'data':{'id':'001'},'evidence':{'id':{'page':1,'quote':'001'}},'review_fields':[]})
        def close(self): pass
    monkeypatch.setattr(adp,'LocalModel',Model)
    monkeypatch.setattr(adp,'refine',lambda model,task,images,rows,result,*args:result)
    monkeypatch.setattr(adp,'generate_questions',lambda *args:{})
    result=adp.run(image,SCHEMA,ocr_pages=[{'pageNumber':1,'text':'ID 001'}])
    assert result['ocr_text']=='ID 001'
    assert result['extracted_fields']=={'id':'001'}
    assert len(calls)==3  # Text extraction, VLM extraction and reconciliation.
    assert any('DOCUMENT OCR:' in prompt and 'ID 001' in prompt for prompt,count in calls)
    assert sum(count>0 for prompt,count in calls)==2
