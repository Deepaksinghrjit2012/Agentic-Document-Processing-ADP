import json
import pytest
from fastapi.testclient import TestClient
import server
from evaluation import generate_questions, extraction_payload

def test_questions_use_one_llm_call_for_all_fields():
    class Model:
        calls=0
        def generate(self,prompt,**kwargs):
            self.calls+=1
            return json.dumps({'id_correct':'Is the extracted id the identifier beside its matching label?',
                               'total_correct':'Is the extracted total numerically equal to the labeled total?'})
    model=Model()
    result=generate_questions(model,{'id':'001','total':'42.50'},{},json.loads)
    assert model.calls==1
    assert len(result)==2
    assert result['id_correct']['type']=='noul'


def test_invalid_batch_questions_use_simple_fallback_without_retry():
    class Model:
        calls=0
        def generate(self,*args,**kwargs):
            self.calls+=1
            return '{}'
    model=Model()
    result=generate_questions(model,{'id':'001'},{},json.loads)
    assert model.calls==1
    assert 'matching label' in result['id_correct']['instructions']
    assert 'leading zeros' in result['id_correct']['instructions']

def test_api_returns_only_fields_and_confidence(monkeypatch):
    result={'ocr_text':'ID 001\nAmount 1250.000000000000000000001',
            'extracted_fields':{'id':'001','amount':'1250.00'},'needs_review':True,
            'evaluation_questions':{'id_correct':{'type':'noul','instructions':'Is id supported?'}}}
    monkeypatch.setattr(server,'run_worker',lambda *args:result)
    c=TestClient(server.app)
    response=c.post('/extract',files={'file':('sample.png',b'fake')},data={'schema':'{}'})
    assert response.status_code==200
    assert response.json()==extraction_payload(result)
    assert set(response.json())=={'extracted_fields'}
    assert response.headers['X-Needs-Review']=='true'
    assert 'X-Evaluation-Questions-URL' not in response.headers
    assert 'X-Diagnostics-URL' not in response.headers


def test_all_field_scores_include_unavailable_and_nested():
    from jev import confidence_payload
    result={'data':{'id':'001','items':[{'amount':'42.50'}]},'review_fields':['items'],
            'jev':{'status':'completed','scores':{'id':0.99}}}
    fields=confidence_payload(result)['sections'][0]['fields']
    assert fields[0]['support_percent']==99
    assert fields[1]['field']=='items.0.amount'
    assert fields[1]['support_probability'] is None
    assert fields[1]['needs_review']


def test_inline_jev_scores_preserve_values():
    result={'ocr_text':'ID 001','extracted_fields':{'id':'001','missing':None},
            'jev':{'scores':{'id':0.99}}}
    assert extraction_payload(result)=={'extracted_fields':{
        'id':{'value':'001','jev_confidence':0.99},
        'missing':{'value':None,'jev_confidence':None}}}


def test_semantic_questions_do_not_require_fixed_formats():
    from evaluation import field_question
    date = field_question(('member_date_of_birth',))
    name = field_question(('plan_member_name',))
    amount = field_question(('total_claim_amount',))
    assert 'same calendar date' in date and 'numeric versus written month' in date
    assert 'form date order' in date
    assert 'ignoring case, spacing and punctuation' in name
    assert 'initials' in name
    assert 'exact numeric amount' in amount and 'without rounding differences' in amount
    assert 'leading zeros' in field_question(('policy_number',))
