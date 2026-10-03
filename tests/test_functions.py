import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_functions_adapter_registers_protected_asgi_route_and_writable_paths(tmp_path):
    env=os.environ.copy()
    env.pop('ADP_REQUEST_TIMEOUT',None)
    env['ADP_LOG_DIR']=str(tmp_path/'logs')
    env['ADP_OUTPUT_DIR']=str(tmp_path/'outputs')
    code='''
import json, os
import function_app
functions=function_app.app.get_functions()
assert len(functions)==1
metadata=functions[0].get_dict_repr()
trigger=next(binding for binding in metadata['bindings'] if binding['type']=='httpTrigger')
assert trigger['authLevel'].value=='function'
assert trigger['route']=='/{*route}'
assert os.environ['ADP_REQUEST_TIMEOUT']=='210'
assert '/extract' in function_app.api.openapi()['paths']
assert '/classify' in function_app.api.openapi()['paths']
print('Functions adapter verified')
'''
    completed=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    assert (tmp_path/'logs/adp.log').exists()


def test_host_config_leaves_http_return_time():
    host=json.loads((ROOT/'host.json').read_text())
    assert host['extensions']['http']['routePrefix']==''
    assert host['functionTimeout']=='00:03:40'
