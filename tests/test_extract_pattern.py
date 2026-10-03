import pytest
from extract_pattern import spotting_page, reconstruct_rows, _format_lines_as_text, tasks_for_schema, grounding_report

def box(text, x, y, right, bottom):
    return text+''.join(f'<|LOC_{n}|>' for n in (x,y,right,y,right,bottom,x,bottom))

def test_coordinate_reading_order_and_horizontal_spacing():
    raw = '\n'.join([box('Contract Number',600,400,900,450), box('00012',600,200,800,250),
                     box('Participant Name',50,400,350,450), box('Jane Doe',50,200,300,250)])
    page = spotting_page(raw, 1000, 1000, 1)
    rows = reconstruct_rows(page)
    assert rows[0].index('Jane Doe') < rows[0].index('00012')
    assert rows[1].index('Participant Name') < rows[1].index('Contract Number')
    assert '   ' in rows[0].split('Jane Doe')[1].split('00012')[0]
    assert 'Page 4\nLine 1 |' in _format_lines_as_text(rows and [rows],4)

def test_incomplete_spotting_is_rejected():
    with pytest.raises(ValueError, match='coordinates'):
        spotting_page(box('Jane',10,10,50,50)+'\nMissing coordinates',1000,1000,1)

SCHEMA = {'document_types':[{'name':'Forms','sub_types':[
    {'name':'A','fields':[{'name':'ID~number','condition':'Only if filled','value_description':'Keep zeros'}]},
    {'name':'B','fields':[{'name':'Reason','allowed_values':['Retired','Other']}]}]}],
    'bulk_rules':[{'bulk_type':'table','table_fields':['Name','ID']} ]}

def test_subtype_sections_keep_original_keys_and_rules():
    tasks = tasks_for_schema(SCHEMA, {'is_bulk_request':True,'sections':[
        {'sub_type':'A','pages':'2-3'}, {'sub_type':'B','pages':'4','has_multi_party_table':True}]},4)
    assert list(tasks[0]['schema']['properties']) == ['ID~number']
    assert 'Only if filled' in tasks[0]['guidance']
    assert 'Keep zeros' in tasks[0]['guidance']
    assert tasks[1]['schema']['properties']['Reason']['enum'] == ['Retired','Other',None]
    assert tasks[1]['schema']['properties']['table_rows']['items']['required'] == ['Name','ID']

def test_page_range_rejects_out_of_document():
    with pytest.raises(ValueError, match='outside document'):
        tasks_for_schema(SCHEMA,{'sections':[{'pages':'2-4'}]},3)

def test_grounding_cannot_match_quote_on_another_page():
    report = grounding_report({'ID':'001'},{'ID':{'page':3,'quote':'001'}},[['001'],['002']],2)
    assert not report['ID']['quote_grounded']
    assert report['ID']['calibrated_confidence'] is None
