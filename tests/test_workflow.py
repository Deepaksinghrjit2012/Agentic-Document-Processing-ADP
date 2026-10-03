import threading
from PIL import Image
from workflow import run_document_graph


def test_ocr_and_vlm_overlap_text_runs_without_waiting_for_vlm_and_join_waits():
    both_started=threading.Barrier(2)
    text_done=threading.Event()
    def ocr(images):
        both_started.wait(timeout=3)
        return [{'pageNumber':1,'text':'ID 001'}],[['ID 001']]
    def visual(task,images):
        both_started.wait(timeout=3)
        assert text_done.wait(timeout=3)
        return {'id':'visual'}
    def text(task,rows):
        assert rows==[['ID 001']]
        text_done.set()
        return {'id':'text'}
    def judge(task,images,rows,first,second):
        assert first=={'id':'text'} and second=={'id':'visual'}
        return {'data':{'id':'001'}}
    result=run_document_graph([Image.new('RGB',(10,10))],[{'start':1,'end':1}],
        ocr,text,visual,judge,lambda task,images,rows,result:result)
    assert result['results']==[{'data':{'id':'001'}}]
    assert 'vlm_extraction' in result['graph_mermaid']


def test_concurrent_sections_keep_order_and_candidates_separate():
    tasks=[{'start':1,'end':1,'id':0},{'start':2,'end':2,'id':1}]
    result=run_document_graph([1,2],tasks,lambda images:([{'pageNumber':1},{'pageNumber':2}],[['A'],['B']]),
        lambda task,rows:task['id'],lambda task,images:task['id']+10,
        lambda task,images,rows,first,second:{'value':(first,second)},
        lambda task,images,rows,result:result)
    assert result['results']==[{'value':(0,10)},{'value':(1,11)}]
