# ADP — Agentic Document Processing

A Python API for classifying and extracting fields from PDFs and images, using an Azure-hosted GPT vision deployment and TypeSafe Jev. LangGraph runs independent work concurrently. Models run through APIs; no GPU or local model downloads are required.

## How it works

**Classification:** GPT transcribes each page in a separate concurrent call. Page text is combined in order and sent to Jev as `state.ocr_text`. Your classification schema is sent unchanged as Jev `questions`; GPT does not generate classification rules. The response includes the document type, Jev confidence, combined OCR and individual page transcripts.

**Extraction:** An OCR/text extraction branch and an independent image extraction branch run concurrently. GPT reconciles their results using the original images. Jev checks field-specific questions and returns per-field confidence. Only low-confidence values are retried using focused image crops, up to two times. All successful retries in a round are evaluated together by Jev.

If you supply OCR from classification, extraction skips its initial full-page OCR calls. The text extraction, image extraction, reconciliation and retry steps remain the same. Crop retries can still perform fresh OCR.

## Local setup

Requires **Python 3.12**, an Azure GPT deployment supporting text/image inputs through the Responses API, and a Jev API key. Use an interactive deployment rather than GlobalBatch.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your endpoint, deployment name and API keys. Alternatively, set environment variables. Keep `.env` private; it is excluded from Git and Azure deployment.

```sh
python server.py
```

Open **http://127.0.0.1:8000/docs** to try both APIs. The homepage provides an extraction form. Progress appears in `logs/adp.log`; cancelling is available through `POST /cancel`. The server accepts one document at a time per worker.

## Schema examples

| Example | Purpose |
|---|---|
| [classification.schema.json](examples/classification.schema.json) | Jev Choice question for health claims, invoices and promissory notes |
| [invoice.schema.json](examples/invoice.schema.json) | Invoice number and total amount extraction |
| [manulife-critical.schema.json](examples/manulife-critical.schema.json) | Critical health-claim fields |

Extraction schemas use JSON Schema. Define labels and the correct person/record in field descriptions. Store identifiers as strings to preserve zeros. State any date or amount normalization explicitly. Classification schemas use Jev's native question format: `document_type` with `type: "choice"`, `instructions`, and `criteria` mapping class names to descriptions/rules/keyword guidance. Include `unclassified` for an unknown option.

## API usage

### Classify a document

POST `/classify` with multipart `file` and `schema`:

```sh
curl -X POST http://127.0.0.1:8000/classify \
  -F 'file=@document.pdf' \
  -F 'schema=<examples/classification.schema.json' \
  -o classification-response.json
```

Response structure (illustrative scores):

```json
{
  "classification": {
    "document_type": "invoice",
    "confidence": 0.95,
    "probabilities": {"invoice": 0.95, "unclassified": 0.05}
  },
  "ocr_text": "Invoice number: INV-123\nTotal: 42.50",
  "ocr_pages": [{"pageNumber": 1, "text": "Invoice number: INV-123\nTotal: 42.50"}]
}
```

The probability map includes all choices in your supplied schema. Invalid or failed Jev classification returns an error rather than a guessed class.

### Extract fields

POST `/extract` with multipart `file`, `schema`, and optional `ocr_pages`:

```sh
curl -X POST http://127.0.0.1:8000/extract \
  -F 'file=@document.pdf' \
  -F 'schema=<examples/invoice.schema.json' \
  -F 'ocr_pages=<classification-response.json'
```

The `ocr_pages` input accepts **the entire classification response** or just its page array. Extraction picks the array internally. Use the same document; page count/order and nonempty text are validated, but the API cannot verify that supplied OCR belongs to the file. Omit `ocr_pages` to run OCR during extraction.

```json
{
  "extracted_fields": {
    "invoice_number": {"value": "INV-123", "jev_confidence": 0.98},
    "total": {"value": 42.50, "jev_confidence": 0.96}
  }
}
```

Extraction returns only fields with their Jev scores. Scores are 0–1; unavailable scores are null. Nested fields have per-value scores. Both APIs return review status in the `X-Needs-Review` header. `/api/classify` and `/api/analyze` are aliases for classification and extraction respectively.

### CLI extraction

```sh
python adp.py document.pdf --schema examples/invoice.schema.json --output outputs/result.json
```

Create the output directory first. The CLI also writes question, confidence and diagnostic sidecars.

## Deploy to Azure Functions

See [the Azure deployment guide](docs/azure-functions.md) for creating a Python 3.12 Flex Consumption app, configuring keys, publishing, and calling hosted endpoints.

The repository includes `function_app.py`, `host.json`, `requirements.txt`, `.funcignore` and a local Functions settings template. Hosted routes require a function key. Keep that key on your website backend. Use remote build so dependencies are installed for Linux rather than copied from a local Mac/Windows environment.

```sh
az login
func azure functionapp publish YOUR_FUNCTION_APP_NAME --build remote
```

No Azure resources are created by local setup. Flex On Demand with Always Ready set to zero has no idle compute charge; storage may still be billed.

## Configuration and limits

`.env.example` lists supported settings. The important defaults are:

- `JEV_THRESHOLD=0.9`: low-confidence extraction retry threshold.
- `JEV_CLASSIFICATION_THRESHOLD=0.9`: classification review threshold.
- `ADP_GRAPH_CONCURRENCY=16` and `ADP_MAX_PARALLEL_REQUESTS=16`: graph and Azure request concurrency; lower these if your deployment throttles.
- `ADP_GENERATION_TIMEOUT=60`: timeout per GPT request.
- `ADP_REQUEST_TIMEOUT=900`: local worker timeout. Azure Functions defaults to 210 seconds to leave time before Azure's 230-second HTTP response limit. Longer jobs need a background queue and polling API.

Inputs support PDFs of **1–20 pages**, up to **30 MB**, or single-frame images. Maximum rendered image size is 20 million pixels per page. Password-protected PDFs requiring a password are rejected. Large documents may exceed memory, transcription budgets or hosting timeouts.

Documents are sent to your Azure GPT deployment and OCR evidence is sent to Jev. Jev confidence measures text support, not guaranteed image accuracy. Both extraction candidates use the same GPT deployment. Original disagreements remain review flags even if retries increase confidence. Validate thresholds on your own labeled documents.

Local logs and crop outputs are excluded from Git. Azure logs/crops use temporary instance storage; they are not durable. The log/cancel endpoints are instance-local and may not target the same worker when Azure scales. LangSmith tracing is disabled.

## Development

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use mocked services and do not require paid GPT/Jev calls.
