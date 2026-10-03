"""Text evidence evaluation with TypeSafe Jev. Never treats scores as image truth."""
import math
import httpx
from azure_model import config
from progress import report
from evaluation import field_targets, field_question


class Jev:
    def __init__(self):
        setting = config()
        self.key = setting('JEV_API_KEY')
        self.endpoint = setting('JEV_ENDPOINT', 'https://api.typesafe.ai/v1/systemone')
        self.model = setting('JEV_MODEL', 'jev-latest')
        self.threshold = float(setting('JEV_THRESHOLD', '0.9'))
        if not 0 <= self.threshold <= 1:
            raise ValueError('JEV_THRESHOLD must be between zero and one')
        if self.endpoint != 'https://api.typesafe.ai/v1/systemone':
            raise ValueError('JEV_ENDPOINT must be the TypeSafe evaluation endpoint')

    def evaluate(self, result, text, paths=None):
        if not self.key:
            return {'status':'disabled', 'scores':{}, 'low_fields':[]}
        paths = list(field_targets(result['data'])) if paths is None else list(paths)
        fields = ['.'.join(str(part) for part in path) for path in paths]
        questions = {f'field_{i}': result.get('field_questions', {}).get('.'.join(str(part) for part in path),
                           {'type':'noul', 'instructions':field_question(path)})
                     for i,path in enumerate(paths)}
        retry_evidence=result.get('retry_evidence',{})
        if retry_evidence:
            for i,field in enumerate(fields):
                question=questions[f'field_{i}']
                questions[f'field_{i}']={**question,'instructions':{
                    'question':question['instructions'],
                    'evidence_source':f'retry_evidence[{__import__("json").dumps(field)}]',
                    'scope':'Evaluate this field only against its own crop OCR text and matching label; other field crops are not evidence for this value.'}}
        if not questions:
            return {'status':'completed', 'scores':{}, 'low_fields':[]}
        report(f'Jev: evaluating {len(fields)} fields')
        try:
            with httpx.Client(timeout=45, follow_redirects=False) as client:
                response = client.post(self.endpoint, headers={'Authorization':'Bearer '+self.key}, json={
                    'model':self.model, 'state':{'ocr_text':text, 'extracted_fields':result['data'],
                    'candidates':{} if retry_evidence else result['candidates'],
                    'retry_evidence':retry_evidence, 'field_definitions':result.get('evaluation_schema', {})}, 'questions':questions})
            if response.status_code != 200:
                raise ValueError(f'HTTP {response.status_code}')
            answers = response.json()['answers']
            scores = {}
            for i, field in enumerate(fields):
                answer = answers[f'field_{i}']
                score = answer['noul']
                if answer['type'] != 'noul' or type(score) not in (int,float) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('Invalid Noul answer')
                scores[field] = score
            return {'status':'completed','scores':scores,
                    'low_fields':sorted({str(path[0]) for path, field in zip(paths,fields) if scores[field] < self.threshold}),
                    'field_paths':dict(zip(fields,paths)),
                    'model':self.model, 'threshold':self.threshold}
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            report('Jev evaluation failed; fields remain flagged for review')
            return {'status':'error','scores':{}, 'low_fields':list(result['data'])}

    def classify(self,text,questions):
        question=questions['document_type']
        if not self.key:
            raise ValueError('Set JEV_API_KEY in .env to classify documents')
        try:
            with httpx.Client(timeout=45,follow_redirects=False) as client:
                response=client.post(self.endpoint,headers={'Authorization':'Bearer '+self.key},json={
                    'model':self.model,'state':{'ocr_text':text},
                    'questions':questions})
            if response.status_code!=200:
                raise RuntimeError(f'Jev classification failed (HTTP {response.status_code})')
            answer=response.json()['answers']['document_type']
            probabilities=answer['probabilities']
            confidence=answer['confidence']
            if (answer['type']!='choice' or answer['choice'] not in question['criteria']
                    or type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1
                    or not isinstance(probabilities,dict) or set(probabilities)!=set(question['criteria'])
                    or any(type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1 for value in probabilities.values())
                    or abs(sum(probabilities.values())-1)>0.01):
                raise ValueError('Invalid Jev classification answer')
            return answer
        except httpx.HTTPError:
            raise RuntimeError('Jev classification connection failed or timed out') from None
        except (ValueError,KeyError,TypeError):
            raise RuntimeError('Jev returned an invalid classification answer') from None


def confidence_payload(result):
    sections = result.get('extractions', [result])
    output = []
    for section in sections:
        evaluation = section.get('jev', {})
        fields = []
        for path in field_targets(section.get('data', {})):
            name = '.'.join(str(part) for part in path)
            score = evaluation.get('scores', {}).get(name)
            fields.append({'field':name, 'path':list(path), 'support_probability':score,
                           'support_percent':round(score*100,2) if score is not None else None,
                           'needs_review':str(path[0]) in section.get('review_fields', []) if path else True})
        output.append({'sub_type':section.get('sub_type'), 'pages':section.get('pages'),
                       'status':evaluation.get('status','unavailable'), 'fields':fields})
    return {'score_type':'noul_yes_probability', 'sections':output}
