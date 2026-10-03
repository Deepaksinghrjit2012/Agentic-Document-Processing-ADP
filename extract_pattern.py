"""Local adaptation of helpers from supplied Extract-main.zip. No cloud SDKs."""
import math
import logging
import re
import statistics
import json
ROW_TOLERANCE = 18.0  # 0.12 inch at 150 DPI

def _word_centroid(polygon: list):
    """Mean of a polygon's corner points: (cx, cy)."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)

def _deskew_anchor(polygon: list, cos_t: float, sin_t: float, cx: float, cy: float):
    """De-rotate a word's centroid back to upright page space.

    DI polygons are in the rotated image frame; `page.angle` tells us the
    clockwise rotation of the content. Rotating each centroid by the inverse
    angle (around the page center, y-down image coordinates) puts every word
    back where it would be on an upright page, so Y-grouping / X-sorting work.
    """
    px, py = _word_centroid(polygon)
    tx, ty = px - cx, py - cy
    return (cx + tx * cos_t + ty * sin_t,
            cy - tx * sin_t + ty * cos_t)

def _reconstruct_page(page: dict, row_tolerance: float = ROW_TOLERANCE) -> list:
    """Group words on a page into rows using Y-coordinate proximity.

    Coordinates are de-skewed by the page rotation angle BEFORE grouping,
    so rotated scans (e.g. a ~90° page) and small skews reconstruct in
    correct reading order. angle == 0 makes the transform a no-op.
    """
    angle  = page.get("angle") or 0.0          # clockwise degrees
    width  = page.get("width") or 0.0
    height = page.get("height") or 0.0
    rad          = math.radians(angle)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    cx, cy       = width / 2.0, height / 2.0

    items: list = []
    for w in page.get("words", []):
        poly = w.get("polygon")
        if not poly:
            continue
        items.append({
            "content": w.get("content", ""),
            "anchor":  _deskew_anchor(poly, cos_t, sin_t, cx, cy),
        })
    for mark in page.get("selectionMarks", []):
        poly = mark.get("polygon")
        if not poly:
            continue
        state = (mark.get("state") or "unselected").upper()
        items.append({
            "content": f"[{state}]",
            "anchor":  _deskew_anchor(poly, cos_t, sin_t, cx, cy),
        })

    if not items:
        return []

    sorted_items = sorted(items, key=lambda w: w["anchor"][1])
    rows = []
    current_row = [sorted_items[0]]
    current_y   = sorted_items[0]["anchor"][1]

    for item in sorted_items[1:]:
        item_y = item["anchor"][1]
        if abs(item_y - current_y) <= row_tolerance:
            current_row.append(item)
        else:
            current_row.sort(key=lambda w: w["anchor"][0])
            rows.append(current_row)
            current_row = [item]
            current_y   = item_y

    if current_row:
        current_row.sort(key=lambda w: w["anchor"][0])
        rows.append(current_row)

    row_texts = []
    for row in rows:
        line_text = "  ".join(w["content"] for w in row)
        if line_text.strip():
            row_texts.append(line_text)
    return row_texts

def _format_lines_as_text(pages_lines: list, start_page_num: int = 1) -> str:
    parts = []
    for offset, line_contents in enumerate(pages_lines):
        page_num = start_page_num + offset
        parts.append(f"Page {page_num}")
        for line_idx, line_content in enumerate(line_contents, start=1):
            parts.append(f"Line {line_idx} | {line_content}")
    return "\n".join(parts)

def _get_fields_for_subtype(schema: dict, sub_type: str) -> list:
    all_sub_types = []
    for dt in schema.get("document_types", []):
        for st in dt.get("sub_types", []):
            name = st.get("name", "") if isinstance(st, dict) else st
            all_sub_types.append(name)
            if name.lower() == sub_type.lower():
                logging.info(f"Exact sub-type match: '{sub_type}' -> '{name}'")
                return st.get("fields", []) if isinstance(st, dict) else []
    for dt in schema.get("document_types", []):
        for st in dt.get("sub_types", []):
            name = st.get("name", "") if isinstance(st, dict) else st
            if name and (name.lower() in sub_type.lower() or sub_type.lower() in name.lower()):
                logging.info(f"Partial sub-type match: '{sub_type}' -> '{name}'")
                return st.get("fields", []) if isinstance(st, dict) else []
    logging.warning(f"No match for sub-type '{sub_type}'. Available: {all_sub_types}")
    return []

def _get_all_schema_fields(schema: dict) -> list:
    all_fields = []
    seen = set()
    for dt in schema.get("document_types", []):
        for st in dt.get("sub_types", []):
            fields = st.get("fields", []) if isinstance(st, dict) else []
            for f in fields:
                name = f.get("name", "") if isinstance(f, dict) else f
                if name and name not in seen:
                    all_fields.append(f)
                    seen.add(name)
    return all_fields

def _find_multi_party_table_rule(schema: dict) -> dict:
    for rule in schema.get("bulk_rules", []):
        bt   = rule.get("bulk_type", "").lower()
        desc = rule.get("description", "").lower()
        if "table" in bt or "table" in desc:
            return rule
    return {}

def _build_fields_prompt(sub_type: str, fields: list) -> str:
    lines = [f"For sub-type '{sub_type}' extract these fields:"]
    for f in fields:
        if isinstance(f, dict):
            name           = f.get("name", "")
            condition      = f.get("condition", "")
            allowed_values = f.get("allowed_values")
            value_desc     = f.get("value_description", "")
        else:
            name, condition, allowed_values, value_desc = f, "", None, ""
        # Field names pass through VERBATIM (including any "~label hint"
        # suffix) — the schema name is the extraction key and response key
        # exactly as given. The hint is consumed only by BM25 label matching.
        line = f"  - {name}"
        if condition:
            line += f"  ({condition})"
        if allowed_values:
            quoted = ", ".join(f'"{v}"' for v in allowed_values)
            line += f"  — allowed values: [{quoted}]"
        lines.append(line)
        if value_desc:
            lines.append(f"      {value_desc}")
    return "\n".join(lines)

# Paddle's spotting output uses eight normalized (0..1000) LOC coordinates
# after each text span. Convert to pixels before using the reference helpers.
SPOT = re.compile(r'([^\n]*?)((?:<\|LOC_\d+\|>){8})')

def spotting_page(raw, width, height, page_number):
    words = []
    for match in SPOT.finditer(raw):
        content = match.group(1).strip()
        coords = [int(v) for v in re.findall(r'<\|LOC_(\d+)\|>', match.group(2))]
        if not content or len(coords) != 8 or any(v > 1000 for v in coords):
            continue
        poly = [v * (width if i % 2 == 0 else height) / 1000 for i, v in enumerate(coords)]
        words.append({'content': content, 'polygon': poly, 'confidence': None})
    remainder = SPOT.sub('', raw).strip()
    if not words or remainder:
        raise ValueError('OCR spotting returned incomplete/unrecognized coordinates; refusing to invent layout')
    angles = [math.degrees(math.atan2(w['polygon'][3]-w['polygon'][1],
                                     w['polygon'][2]-w['polygon'][0])) for w in words]
    return {'pageNumber': page_number, 'width': width, 'height': height, 'unit': 'pixel',
            'angle': statistics.median(angles), 'words': words, 'selectionMarks': [],
            'selection_marks_available': False}

def reconstruct_rows(page):
    # Follow reference deskew/Y-grouping/X-sorting, retaining horizontal gaps.
    angle = math.radians(page.get('angle') or 0)
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    cx, cy = page['width']/2, page['height']/2
    items = []
    for word in page['words']:
        x, y = _deskew_anchor(word['polygon'], cos_t, sin_t, cx, cy)
        items.append((x, y, word['content'], word['polygon']))
    groups = []
    for item in sorted(items, key=lambda v: v[1]):
        if not groups or abs(item[1] - groups[-1][0][1]) > ROW_TOLERANCE:
            groups.append([item])
        else:
            groups[-1].append(item)
    rows = []
    for group in groups:
        chunks, end = [], 0
        for x, y, content, poly in sorted(group):
            # Use the de-rotated left edge rather than centroids for spacing.
            left = min(_deskew_anchor([poly[i], poly[i+1]], cos_t, sin_t, cx, cy)[0]
                       for i in range(0,8,2))
            column = max(0, round(left / page['width'] * 160))
            gap = max(2 if chunks else 0, column-end)
            chunks.append(' '*gap + content)
            end += gap + len(content)
        rows.append(''.join(chunks).rstrip())
    return rows

FORM_GUIDANCE = '''DOCUMENT FORMAT: Page N followed by Line N | model-recognized text line.
OCR lines are GPT transcriptions of page images, without measured coordinates.
Use labels and nearby text; do not infer exact horizontal alignment from line order.
Line numbering restarts on each page. These prefixes are navigation, never data.
For two-line forms, values often appear immediately ABOVE their labels.
Match horizontal position: left value to left label, right value to right label.
Do not swap adjacent fields. Check same-line label:value patterns too.
Never return a printed label as a filled value. Preserve original strings and leading zeros.
Ignore footer/legal boilerplate. Respect field conditions and value descriptions.
For allowed_values fields, read selected checkbox labels and return the matching allowed
value. If multiple are selected, use the first in reading order; none or unclear means null.
OCR selection-mark states are unavailable unless explicitly supplied; never infer a
selected state from a label's presence alone. Flag uncertain checkbox-backed fields.
For tables use headers to match columns, extract every data row, exclude headers,
and do not mix records between rows. Use null for missing cells.
'''

def tasks_for_schema(schema, classification, total_pages):
    if not isinstance(schema, dict) or not isinstance(classification, dict):
        raise ValueError('Schema and classification must be objects')
    reference = 'document_types' in schema
    subtypes = classification.get('document_sub_type', [])
    if not isinstance(subtypes, list) or not all(isinstance(s, str) for s in subtypes):
        raise ValueError('document_sub_type must be an array of strings')
    sections = classification.get('sections', [])
    if not isinstance(sections, list) or not all(isinstance(s, dict) for s in sections):
        raise ValueError('sections must be an array of objects')
    bulk = classification.get('is_bulk_request', False)
    table_bulk = 'table' in str(classification.get('bulk_type', '')).lower()
    specs = sections if sections else [{'sub_type': sub, 'pages': f'1-{total_pages}',
                                       'has_multi_party_table': bool(bulk and table_bulk)}
                                      for sub in (subtypes if bulk and subtypes else [subtypes[0] if subtypes else ''])]
    tasks = []
    for spec in specs:
        rng = str(spec.get('pages', f'1-{total_pages}'))
        if not re.fullmatch(r'\d+(?:-\d+)?', rng):
            raise ValueError(f'Invalid page range: {rng}')
        nums = [int(n) for n in rng.split('-')]
        start, end = nums[0], nums[-1]
        if not 1 <= start <= end <= total_pages:
            raise ValueError(f'Page range outside document: {rng}')
        subtype = spec.get('sub_type', '')
        if not isinstance(subtype, str):
            raise ValueError('sub_type must be a string')
        guidance = FORM_GUIDANCE
        if reference:
            fields = _get_fields_for_subtype(schema, subtype) or _get_all_schema_fields(schema)
            if not fields:
                raise ValueError(f'No schema fields for subtype {subtype}')
            props = {}
            for field in fields:
                rec = field if isinstance(field, dict) else {'name': field}
                name = rec.get('name')
                if not isinstance(name, str) or not name or name in props:
                    raise ValueError('Field names must be unique nonempty strings')
                props[name] = {'type': ['string', 'null']}
                if 'allowed_values' in rec:
                    values = rec['allowed_values']
                    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                        raise ValueError('allowed_values must be an array of strings')
                    props[name]['enum'] = values + [None]
            effective = {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}
            guidance += '\n' + _build_fields_prompt(subtype, fields)
        else:
            import copy
            effective = copy.deepcopy(schema)
        if spec.get('has_multi_party_table'):
            rule = _find_multi_party_table_rule(schema)
            columns = rule.get('table_fields', [])
            if not columns or not all(isinstance(c, str) and c for c in columns):
                raise ValueError('Table extraction requires bulk_rules table_fields')
            if 'table_rows' in effective.get('properties', {}):
                raise ValueError('table_rows conflicts with a schema field')
            effective.setdefault('properties', {})['table_rows'] = {
                'type': 'array', 'items': {'type': 'object', 'properties':
                {c: {'type': ['string', 'null']} for c in columns},
                'required': columns, 'additionalProperties': False}}
            effective.setdefault('required', []).append('table_rows')
            guidance += '\nTABLE COLUMNS: ' + json.dumps(columns)
        tasks.append({'sub_type': subtype, 'pages': rng, 'start': start, 'end': end,
                      'schema': effective, 'guidance': guidance, 'has_table': bool(spec.get('has_multi_party_table'))})
    return tasks


def grounding_report(data, evidence, rows, start_page):
    # Deterministic quote lookup: never claim calibrated likelihood from it.
    report = {}
    for field, value in data.items():
        ev = evidence.get(field, {})
        page = ev.get('page') if isinstance(ev, dict) else None
        quote = ev.get('quote') if isinstance(ev, dict) else None
        lines = rows[page-start_page] if type(page) is int and start_page <= page < start_page+len(rows) else []
        normalize = lambda s: ' '.join(str(s).split()).casefold()
        matched = [i+1 for i, line in enumerate(lines) if isinstance(quote, str) and quote.strip()
                   and normalize(quote) in normalize(line)]
        report[field] = {'quote_grounded': bool(matched), 'page': page, 'lines': matched,
                         'ocr_confidence': None, 'logprob_confidence': None, 'calibrated_confidence': None}
    return report
