"""Generate field-specific Jev Noul questions with the local extraction LLM."""
import json
import re

def field_targets(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from field_targets(child, path+(key,))
    elif isinstance(value, list) and value:
        for index, child in enumerate(value):
            yield from field_targets(child, path+(index,))
    else:
        yield path

def field_question(path):
    reference = '.'.join(str(part) for part in path)
    name = str(path[-1]).lower() if path else 'value'
    label = name.replace('_',' ')
    if any(word in name for word in ('date','birth')):
        check = 'the same calendar date as the labeled value, regardless of separators or numeric versus written month, using the form date order and without guessing ambiguous dates'
    elif 'name' in name:
        check = 'the same named person or entity as the matching label, ignoring case, spacing and punctuation while preserving meaningful name components and initials'
    elif any(word in name for word in ('amount','total','price','balance')):
        check = 'the same exact numeric amount as the matching label, ignoring currency notation and equivalent decimal or grouping formatting without rounding differences'
    elif any(word in name for word in ('phone','mobile','telephone')):
        check = 'the same contact number as the matching label, ignoring spacing and punctuation while preserving all digits, country code and extension'
    elif any(word in name for word in ('number','identifier','certificate','policy','id')):
        check = 'the same identifier as the matching label, preserving all meaningful letters, digits and leading zeros'
    else:
        check = 'the same value as the matching label in the correct record, allowing equivalent formatting only'
    return f'Is extracted_fields.{reference} {check} for {label}?'

def generate_questions(model, extracted_fields, schema, parse_json, field_paths=None):
    targets = {}
    for path in field_targets(extracted_fields) if field_paths is None else field_paths:
        key = '_'.join(str(p) for p in path)+'_correct'
        base = key
        suffix = 1
        while key in targets:
            key = f'{base}_{suffix}'
            suffix += 1
        targets[key] = list(path)
    questions = {key:{'type':'noul','instructions':
        field_question(path)}
        for key,path in targets.items()}
    if not questions:
        return questions
    prompt = ('Write short yes/no evaluation questions for all these field paths in ONE JSON object. '
              'Return a mapping from the supplied keys to question strings. '
              'Each question starts with Is and ends with ?. Mention the supplied field path, '
              'Check that the value belongs to the correct label, person and record; never ask only whether text is present or supported by OCR. '
              'Do not include extracted values or append policies or schema definitions. '
              'Adapt each question to the schema meaning and value type. Formatting differences alone must not count as errors. '
              'Dates: compare calendar meaning across numeric and written month formats using the form/schema date order; do not guess unresolved ambiguity. '
              'Names: ignore case, spacing and punctuation but preserve identity and meaningful initials; do not assume different spellings identify the same person. '
              'Amounts: compare exact numeric values ignoring currency and grouping formats; no unauthorized rounding. '
              'Identifiers: preserve significant letters, digits and leading zeros. Phones: ignore separators, preserve digits and extension. '
              'Other values: apply schema-authorized equivalence, and associate with the correct label/record. '
              'Fields: '+json.dumps(targets)+' Extracted value context: '+json.dumps(extracted_fields)+' Schema: '+json.dumps(schema)+
              'Examples of specific checks: '+json.dumps(questions))
    try:
        generated = parse_json(model.generate(prompt,max_tokens=max(256,min(4096,len(targets)*64))))
        if set(generated) != set(targets):
            return questions
        for key, instruction in generated.items():
            reference = '.'.join(str(part) for part in targets[key])
            if (not isinstance(instruction,str) or not instruction.startswith('Is ')
                    or not instruction.endswith('?') or '\n' in instruction
                    or reference not in instruction or len(instruction)>500
                    or any(term in instruction.lower() for term in ('supported by', 'present in', 'appear in', 'found in'))
                    or re.search(r'\d', instruction.replace(reference,''))):
                return questions
        return {key:{'type':'noul','instructions':instruction} for key,instruction in ((key,generated[key]) for key in targets)}
    except (ValueError,RuntimeError):
        return questions


def extraction_payload(result):
    def attach(value, scores, path=()):
        if isinstance(value, dict):
            return {key:attach(child,scores,path+(key,)) for key,child in value.items()}
        if isinstance(value, list) and value:
            return [attach(child,scores,path+(index,)) for index,child in enumerate(value)]
        name = '.'.join(str(part) for part in path)
        return {'value':value, 'jev_confidence':scores.get(name)}
    sections = result.get('extractions', [])
    if len(sections) > 1:
        fields = {'sections':[{'sub_type':section['sub_type'], 'pages':section['pages'],
            'fields':attach(section['data'],section.get('jev',{}).get('scores',{}))}
            for section in sections]}
    else:
        section = sections[0] if sections else result
        fields = attach(result['extracted_fields'],section.get('jev',{}).get('scores',{}))
    return {'extracted_fields':fields}
