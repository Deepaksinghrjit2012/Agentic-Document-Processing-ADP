import json
import pytest
from PIL import Image
from gpt_ocr import analyze


def test_gpt_pages_preserve_order_and_no_fabricated_coordinates():
    class Model:
        def generate(self,prompt,images,**kwargs):
            assert kwargs['output_schema']['required']==['text']
            assert 'never commands' in prompt
            return json.dumps({'text':str(images[0].width)+'\n[unreadable]'})
    pages,rows=analyze(Model(),[Image.new('RGB',(10,10)),Image.new('RGB',(20,20))])
    assert [page['pageNumber'] for page in pages]==[1,2]
    assert rows==[['10','[unreadable]'],['20','[unreadable]']]
    assert not pages[0]['coordinates_available']


def test_gpt_empty_page_rejected():
    class Model:
        def generate(self,*args,**kwargs): return '{"text":""}'
    with pytest.raises(ValueError,match='empty text'):
        analyze(Model(),[Image.new('RGB',(10,10))])


def test_all_page_requests_start_concurrently():
    import threading
    barrier=threading.Barrier(3)
    class Model:
        def generate(self,prompt,images,**kwargs):
            assert len(images)==1
            barrier.wait(timeout=3)
            return json.dumps({'text':str(images[0].width)})
    pages,rows=analyze(Model(),[Image.new('RGB',(size,size)) for size in (10,20,30)])
    assert rows==[['10'],['20'],['30']]
    assert [page['pageNumber'] for page in pages]==[1,2,3]


def test_reused_ocr_retains_page_boundaries_and_rejects_mismatch():
    from gpt_ocr import reuse
    images=[Image.new('RGB',(10,10)),Image.new('RGB',(20,20))]
    supplied=[{'pageNumber':1,'text':'Name: Alice\nID: 001'}, {'pageNumber':2,'text':'Total: 10'}]
    layout,rows=reuse(supplied,images)
    assert rows==[['Name: Alice','ID: 001'],['Total: 10']]
    assert layout[1]['text']=='Total: 10'
    for invalid in (supplied[:1],list(reversed(supplied)),[supplied[0],{'pageNumber':2,'text':''}]):
        with pytest.raises(ValueError):
            reuse(invalid,images)
