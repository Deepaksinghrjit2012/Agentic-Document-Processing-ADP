import copy
from PIL import Image
import pytest
import feedback


def initial():
    return {'data':{'id':'14','name':'Accepted Name'},'evidence':{},'candidates':{},
            'review_fields':['id'],'schema_errors':[],'disagreements':['id'],'needs_review':True}


def test_only_low_field_retried_and_good_value_preserved(monkeypatch):
    class Evaluator:
        key='test'
        threshold=0.9
        def evaluate(self,result,text,paths=None):
            if paths is None:
                return {'status':'completed','scores':{'id':0.3,'name':0.99},'low_fields':['id']}
            assert paths==[('id',)]
            return {'status':'completed','scores':{'id':0.98},'low_fields':[]}
    monkeypatch.setattr(feedback,'Jev',Evaluator)
    calls=[]
    def retry(*args):
        path=args[4]
        calls.append(path)
        return {'path':list(path),'field':'id','value':'1444','ocr_text':'No. 1444',
                'candidates':{},'evidence':{'page':1,'quote':'1444'},'review':False}
    monkeypatch.setattr(feedback,'retry_field',retry)
    source=initial()
    result=feedback.refine(None,{'start':1,'schema':{}},[],[['No.14']],source,None,None)
    assert calls==[('id',)]
    assert result['data']=={'id':'1444','name':'Accepted Name'}
    assert result['jev']['scores']['name']==0.99
    assert source['data']['id']=='14'
    assert result['feedback_attempts'][0]['data']['id']=='14'
    assert result['review_fields']==['id']


def test_low_confidence_is_bounded_at_two_attempts(monkeypatch):
    class Evaluator:
        key='test'
        def evaluate(self,result,text,paths=None):
            return {'status':'completed','scores':{'id':0.3,'name':0.99} if paths is None else {'id':0.3},'low_fields':['id']}
    monkeypatch.setattr(feedback,'Jev',Evaluator)
    calls=[]
    def retry(*args):
        calls.append(args[4])
        return {'path':['id'],'field':'id','value':'14','ocr_text':'No.14','candidates':{},'evidence':{},'review':True}
    monkeypatch.setattr(feedback,'retry_field',retry)
    result=feedback.refine(None,{'start':1,'schema':{}},[],[['No.14']],initial(),None,None)
    assert len(calls)==2
    assert len(result['feedback_attempts'])==3
    assert result['needs_review']


def test_evaluation_error_stops_retries(monkeypatch):
    class Evaluator:
        key='test'
        def evaluate(self,*args,**kwargs):
            return {'status':'error','scores':{},'low_fields':['id','name']}
    monkeypatch.setattr(feedback,'Jev',Evaluator)
    result=feedback.refine(None,{'start':1},[],[['No.14']],initial(),None,None)
    assert result['needs_review']
    assert len(result['feedback_attempts'])==1


def test_crop_padding_clamps_and_invalid_box_rejected():
    image=Image.new('RGB',(1000,2000))
    assert feedback.padded_box(image,[0,0,100,100])==(0,0,135,250)
    for box in ([100,100,50,50],[-1,0,100,100],[0,0,1,1],[0,0,float('nan'),100]):
        with pytest.raises(ValueError): feedback.padded_box(image,box)


def test_unreliable_locator_falls_back_to_evidence_page():
    class Model:
        def generate(self,*args,**kwargs):
            return '{"page":null,"box":null,"reliable":false}'
    source=initial()
    source['evidence']={'id':{'page':3}}
    images=[Image.new('RGB',(10,10)),Image.new('RGB',(20,20))]
    crop,region=feedback.locate_field(Model(),{'start':2,'end':3,'schema':{}},images,source,('id',))
    assert region['fallback'] and region['pages']==[3]
    assert crop[0].width==20


def test_nested_leaf_retry_and_lower_score_preserves_value(monkeypatch):
    class Evaluator:
        key='test'
        def evaluate(self,result,text,paths=None):
            if paths is None:
                return {'status':'completed','scores':{'items.0.amount':0.7,'items.1.amount':0.99},'low_fields':['items']}
            assert paths==[('items',0,'amount')]
            return {'status':'completed','scores':{'items.0.amount':0.4},'low_fields':['items']}
    monkeypatch.setattr(feedback,'Jev',Evaluator)
    def retry(*args):
        return {'path':['items',0,'amount'],'field':'items.0.amount','value':'999.00',
                'ocr_text':'Amount 999.00','candidates':{},'evidence':{},'review':False}
    monkeypatch.setattr(feedback,'retry_field',retry)
    source={**initial(),'data':{'items':[{'amount':'42.50'},{'amount':'10.00'}]}}
    result=feedback.refine(None,{'start':1,'schema':{}},[],[['Amount']],source,None,None)
    assert result['data']==source['data']
    assert result['jev']['scores']['items.1.amount']==0.99
    assert not result['feedback_attempts'][1]['fields'][0]['accepted']


def test_multiple_retried_fields_use_one_jev_call_with_own_evidence(monkeypatch):
    calls=[]
    class Evaluator:
        key='test'
        threshold=0.9
        def evaluate(self,result,text,paths=None):
            calls.append(paths)
            if paths is None:
                return {'status':'completed','scores':{'id':0.3,'name':0.4,'good':0.99},'low_fields':['id','name']}
            assert set(paths)=={('id',),('name',)}
            assert result['retry_evidence']['id']['ocr_text']=='ID crop'
            assert result['retry_evidence']['name']['ocr_text']=='Name crop'
            assert result['data']['good']=='Keep'
            return {'status':'completed','scores':{'id':0.98,'name':0.99},'low_fields':[]}
    monkeypatch.setattr(feedback,'Jev',Evaluator)
    def retry(*args):
        field=args[4][0]
        return {'path':[field],'field':field,'value':'001' if field=='id' else 'John',
                'ocr_text':'ID crop' if field=='id' else 'Name crop','candidates':{},'evidence':{},'review':False}
    monkeypatch.setattr(feedback,'retry_field',retry)
    source={**initial(),'data':{'id':'14','name':'J','good':'Keep'}}
    result=feedback.refine(None,{'start':1,'schema':{}},[],[['OCR']],source,None,None)
    assert len(calls)==2  # One initial call and one call for the entire retry round.
    assert result['data']=={'id':'001','name':'John','good':'Keep'}
    assert result['jev']['scores']['good']==0.99
