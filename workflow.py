"""LangGraph orchestration; independent branches overlap, joins preserve checks."""
import os
import time
from typing import Annotated, TypedDict
# Graph state includes document data; do not export it to hosted tracing.
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
from langgraph.graph import StateGraph, START, END
from azure_model import config
from progress import report


def merge(left, right):
    return {**left, **right}


class MapState(TypedDict):
    results: Annotated[dict, merge]


def parallel_map(fn, items):
    """Run independent LangGraph nodes concurrently, returning input order."""
    items=list(items)
    if not items:
        return []
    graph=StateGraph(MapState)
    for index,item in enumerate(items):
        def node(state,index=index,item=item):
            return {'results':{index:fn(item)}}
        name=f'item_{index}'
        graph.add_node(name,node)
        graph.add_edge(START,name)
        graph.add_edge(name,END)
    result=graph.compile().invoke({'results':{}},config={'max_concurrency':max(2,min(len(items),int(config()('ADP_GRAPH_CONCURRENCY','16'))))})
    return [result['results'][i] for i in range(len(items))]


class DocumentState(TypedDict):
    layout: list
    rows: list
    text_candidates: dict
    visual_candidates: dict
    reconciled: Annotated[dict, merge]
    results: Annotated[dict, merge]
    timings: Annotated[dict, merge]


def run_document_graph(images,tasks,ocr,text_extract,visual_extract,judge,feedback):
    graph=StateGraph(DocumentState)
    def ocr_text(state):
        started=time.monotonic()
        report('LangGraph: starting concurrent page OCR and independent VLM branch')
        layout,rows=ocr(images)
        ocr_seconds=time.monotonic()-started
        text_started=time.monotonic()
        candidates=parallel_map(lambda item:text_extract(item[1],rows),enumerate(tasks))
        return {'layout':layout,'rows':rows,
                'text_candidates':dict(enumerate(candidates)),
                'timings':{'ocr_seconds':round(ocr_seconds,3),'text_extraction_seconds':round(time.monotonic()-text_started,3),'ocr_text_branch_seconds':round(time.monotonic()-started,3)}}
    def vlm(state):
        started=time.monotonic()
        candidates=parallel_map(lambda item:visual_extract(item[1],images),enumerate(tasks))
        return {'visual_candidates':dict(enumerate(candidates)),
                'timings':{'vlm_branch_seconds':round(time.monotonic()-started,3)}}
    graph.add_node('ocr_and_text_extraction',ocr_text)
    graph.add_node('vlm_extraction',vlm)
    graph.add_edge(START,'ocr_and_text_extraction')
    graph.add_edge(START,'vlm_extraction')
    for index,task in enumerate(tasks):
        def reconcile_node(state,index=index,task=task):
            result=judge(task,images,state['rows'],state['text_candidates'][index],state['visual_candidates'][index])
            return {'reconciled':{index:result}}
        def feedback_node(state,index=index,task=task):
            return {'results':{index:feedback(task,images,state['rows'],state['reconciled'][index])}}
        judge_name=f'reconcile_section_{index+1}'
        feedback_name=f'jev_and_retries_section_{index+1}'
        graph.add_node(judge_name,reconcile_node)
        graph.add_node(feedback_name,feedback_node)
        graph.add_edge(['ocr_and_text_extraction','vlm_extraction'],judge_name)
        graph.add_edge(judge_name,feedback_name)
        graph.add_edge(feedback_name,END)
    concurrency=int(config()('ADP_GRAPH_CONCURRENCY','16'))
    if concurrency<2:
        raise ValueError('ADP_GRAPH_CONCURRENCY must be at least two')
    compiled=graph.compile()
    result=compiled.invoke({'reconciled':{},'results':{},'timings':{}},
                           config={'max_concurrency':concurrency})
    result['results']=[result['results'][i] for i in range(len(tasks))]
    result['graph_mermaid']=compiled.get_graph().draw_mermaid()
    return result
