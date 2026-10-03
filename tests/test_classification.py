import json
import pytest
from fastapi.testclient import TestClient
import classification as service
import server

SCHEMA={'document_type':{'type':'choice','instructions':'Which class fits ocr_text?',
    'criteria':{'invoice':'Invoice with amount due','unclassified':'No matching rule'}}}

def test_schema_is_native_jev_question():
    assert service.validate_questions(SCHEMA) is SCHEMA['document_type']
    for field in ('type','instructions','criteria'):
        question=dict(SCHEMA['document_type']); question.pop(field)
        with pytest.raises(ValueError):
            service.validate_questions({'document_type':question})
    with pytest.raises(ValueError):
        service.validate_questions({'document_types':[]})

def test_gpt_only_does_ocr_and_jev_receives_original_questions(monkeypatch):
    class Model:
        def close(self): pass
        def generate(self,*args,**kwargs):
            raise AssertionError('Must not generate classification questions')
    class Evaluator:
        key='fake'
        def classify(self,text,questions):
            assert text=='Invoice No: secret'
            assert questions is SCHEMA
            return {'choice':'unclassified','confidence':0.8,'probabilities':{'invoice':0.2,'unclassified':0.8}}
    monkeypatch.setattr(service,'AzureModel',Model)
    monkeypatch.setattr(service,'Jev',Evaluator)
    monkeypatch.setattr(service,'pages',lambda path:[object()])
    monkeypatch.setattr(service,'analyze',lambda model,images:([{'pageNumber':1,'text':'Invoice No: secret'}],[]))
    result=service.run('fake.pdf',SCHEMA)
    assert result['classification']['document_type']=='unclassified'
    assert result['needs_review']

def test_jev_request_preserves_questions_and_ocr_only_state(monkeypatch):
    import jev
    class Client:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def post(self,endpoint,headers,json):
            assert json['state']=={'ocr_text':'sample OCR'}
            assert json['questions'] is SCHEMA
            class Response:
                status_code=200
                def json(self):
                    return {'answers':{'document_type':{'type':'choice','choice':'invoice','confidence':0.95,
                        'probabilities':{'invoice':0.95,'unclassified':0.05}}}}
            return Response()
    monkeypatch.setattr(jev.httpx,'Client',Client)
    evaluator=jev.Jev(); evaluator.key='fake'
    assert evaluator.classify('sample OCR',SCHEMA)['choice']=='invoice'

def test_classification_api_returns_jev_result_and_reusable_ocr(monkeypatch):
    result={'classification':{'document_type':'invoice','confidence':0.95,'probabilities':{'invoice':0.95,'unclassified':0.05}},'needs_review':False,'ocr_text':'private','ocr_pages':[{'pageNumber':1,'text':'private'}],'question':{}}
    def worker(path,schema,unused,mode):
        assert mode=='classify' and schema==SCHEMA
        return result
    monkeypatch.setattr(server,'run_worker',worker)
    response=TestClient(server.app).post('/classify',files={'file':('doc.png',b'image')},data={'schema':json.dumps(SCHEMA)})
    assert response.status_code==200
    assert response.json()=={key:result[key] for key in ('classification','ocr_text','ocr_pages')}
    assert response.headers['X-Needs-Review']=='false'


@pytest.mark.parametrize('whole_response',[False,True])
def test_extraction_api_passes_supplied_ocr_to_worker(monkeypatch,whole_response):
    supplied=[{'pageNumber':1,'text':'ID: 001'}]
    def worker(path,schema,classification,ocr_pages=None):
        assert ocr_pages==supplied
        return {'extracted_fields':{'id':'001'},'data':{'id':'001'},'needs_review':False}
    monkeypatch.setattr(server,'run_worker',worker)
    response=TestClient(server.app).post('/extract',files={'file':('doc.png',b'image')},
        data={'schema':'{"type":"object"}','ocr_pages':json.dumps({'classification':{'document_type':'invoice'},'ocr_text':'combined text ignored','ocr_pages':supplied} if whole_response else supplied)})
    assert response.status_code==200
    assert response.json()['extracted_fields']['id']['value']=='001'


@pytest.mark.parametrize('value',[{'ocr_text':'combined only'},{'ocr_pages':None},{'ocr_pages':'invalid'}])
def test_extraction_rejects_response_without_page_array(monkeypatch,value):
    def worker(*args,**kwargs):
        raise AssertionError('Malformed OCR must not start inference')
    monkeypatch.setattr(server,'run_worker',worker)
    response=TestClient(server.app).post('/extract',files={'file':('doc.png',b'image')},
        data={'schema':'{"type":"object"}','ocr_pages':json.dumps(value)})
    assert response.status_code==422
    assert 'containing ocr_pages' in response.json()['detail']
