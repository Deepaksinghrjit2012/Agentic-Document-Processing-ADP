import httpx
import pytest
from PIL import Image
import azure_model


def test_text_image_response_and_incomplete_rejected(monkeypatch):
    monkeypatch.setattr(azure_model, 'config', lambda: lambda name, default='': {
        'AZURE_OPENAI_RESPONSES_ENDPOINT':'https://example.azure.com/openai/v1/responses',
        'AZURE_OPENAI_API_KEY':'test-key'}.get(name, default))
    model = azure_model.AzureModel()
    model.client.close()
    def respond(request):
        import json
        body = json.loads(request.content)
        assert body['model'] == 'gpt-5-mini-1'
        assert body['store'] is False
        assert body['input'][0]['content'][1]['image_url'].startswith('data:image/png;base64,')
        assert request.headers['api-key'] == 'test-key'
        return httpx.Response(200, json={'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'{"id":"001"}'}]}]})
    model.client = httpx.Client(headers={'api-key':'test-key'}, transport=httpx.MockTransport(respond))
    assert model.generate('extract', [Image.new('RGB',(10,10))]) == '{"id":"001"}'
    model.client.close()
    model.client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200,json={'status':'incomplete'})))
    with pytest.raises(RuntimeError,match='incomplete'):
        model.generate('extract')
    model.close()


def test_throttled_request_retries_same_payload(monkeypatch):
    monkeypatch.setattr(azure_model,'config',lambda:lambda name,default='':{
        'AZURE_OPENAI_RESPONSES_ENDPOINT':'https://example.azure.com/openai/v1/responses',
        'AZURE_OPENAI_API_KEY':'test'}.get(name,default))
    monkeypatch.setattr(azure_model.time,'sleep',lambda delay:None)
    model=azure_model.AzureModel()
    model.client.close()
    calls=[]
    def respond(request):
        calls.append(request.content)
        if len(calls)==1:
            return httpx.Response(429,headers={'Retry-After':'1'})
        return httpx.Response(200,json={'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'OK'}]}]})
    model.client=httpx.Client(transport=httpx.MockTransport(respond))
    try:
        assert model.generate('test')=='OK'
        assert len(calls)==2 and calls[0]==calls[1]
    finally:
        model.close()
