"""GPT OCR with caller-supplied Jev classification questions."""
import time
from azure_model import AzureModel, config
from adp import pages
from gpt_ocr import analyze
from jev import Jev
from progress import report

UNKNOWN='unclassified'


def validate_questions(schema):
    if not isinstance(schema,dict) or set(schema) != {'document_type'}:
        raise ValueError('Classification schema must be a Jev questions object with one document_type question')
    question=schema['document_type']
    if not isinstance(question,dict) or question.get('type')!='choice':
        raise ValueError('document_type must be a Jev choice question')
    instructions=question.get('instructions')
    if not ((isinstance(instructions,str) and instructions.strip()) or
            (isinstance(instructions,dict) and instructions)):
        raise ValueError('document_type requires nonempty instructions')
    criteria=question.get('criteria')
    if (not isinstance(criteria,dict) or not 2 <= len(criteria) <= 255
            or any(not isinstance(name,str) or not name.strip() for name in criteria)
            or any(not isinstance(value,str) or not value.strip() for value in criteria.values())):
        raise ValueError('document_type requires 2–255 named criteria with nonempty string descriptions')
    return question


def run(document,schema,device='cpu'):
    question=validate_questions(schema)
    threshold=float(config()('JEV_CLASSIFICATION_THRESHOLD','0.9'))
    if not 0 <= threshold <= 1:
        raise ValueError('JEV_CLASSIFICATION_THRESHOLD must be between zero and one')
    evaluator=Jev()
    if not evaluator.key:
        raise ValueError('Set JEV_API_KEY in .env to classify documents')
    images=pages(document)
    report(f'Classification prepared: {len(images)} pages, {len(question["criteria"])} choices')
    started=time.monotonic()
    model=AzureModel()
    try:
        layout_rows=analyze(model,images)
    finally:
        model.close()
    layout,_=layout_rows
    text='\n\n'.join(page['text'] for page in layout)
    report('Classification: Jev selecting document type from OCR')
    answer=evaluator.classify(text,schema)
    result={'classification':{'document_type':answer['choice'], 'confidence':answer['confidence'],
                              'probabilities':answer['probabilities']},
            'needs_review':answer['choice']==UNKNOWN or answer['confidence']<threshold,
            'ocr_text':text,'ocr_pages':[{'pageNumber':page['pageNumber'],'text':page['text']} for page in layout],
            'question':question,'total_seconds':round(time.monotonic()-started,3)}
    report(f'Classification completed in {result["total_seconds"]}s; needs_review={result["needs_review"]}')
    return result
