# Deploy ADP to Azure Functions

The repository runs as a Python v2 Azure Functions ASGI app. The `/classify`, `/extract`, `/api/classify`, and `/api/analyze` routes retain their existing request/response formats, including reuse of the whole classification response in extraction's `ocr_pages` field. No GPU or local model downloads are required.

## Create the app

In Azure Portal, create a **Function App**, using **Flex Consumption**, **Linux**, **Python 3.12**, and Functions runtime **4**. Select a region with Flex availability. Start with **4096 MB** if you expect larger PDFs; rendered images and parallel vision requests can consume significant memory. Smaller documents can use 2048 MB after measuring peak memory. Keep **Always Ready instances at 0** for scale-to-zero billing. Storage is billed separately.

Set HTTP concurrency to **1** and Python worker process count to **1**. The application processes one document at a time per Python worker; Azure can scale to additional instances. The `/logs` and `/cancel` routes are instance-local, so they are not reliable across scaled instances. They remain useful locally. Long-running production workflows should use a durable job/status API.

## Configure application settings

In the Function App's **Settings → Environment variables**, add the following. Copy secrets directly from your local `.env` into Azure settings; never deploy `.env` or place keys in your website code.

| Setting | Value |
|---|---|
| `AZURE_OPENAI_RESPONSES_ENDPOINT` | Your existing Azure Responses endpoint |
| `AZURE_OPENAI_DEPLOYMENT` | Your deployment name, currently `gpt-5-mini-1` |
| `AZURE_OPENAI_API_KEY` | Your GPT key |
| `JEV_API_KEY` | Your Jev key |
| `JEV_ENDPOINT` | `https://api.typesafe.ai/v1/systemone` |
| `JEV_MODEL` | `jev-latest` |
| `FUNCTIONS_WORKER_PROCESS_COUNT` | `1` |
| `ADP_REQUEST_TIMEOUT` | `210` |
| `ADP_MAX_PARALLEL_REQUESTS` | `16` (lower if Azure throttles) |
| `ADP_GRAPH_CONCURRENCY` | `16` |

Keep platform-generated storage and runtime settings. Optional `JEV_THRESHOLD` and `JEV_CLASSIFICATION_THRESHOLD` default to `0.9`. The entry point sets writable temporary log/crop paths automatically. Temporary files are instance-local and are not durable storage.

## Publish

Install Azure Functions Core Tools v4 and Azure CLI, then run from this repository:

```sh
az login
func azure functionapp publish YOUR_FUNCTION_APP_NAME --build remote
```

Remote build installs the Linux dependencies in `requirements.txt`; do not upload your Mac `.venv`. `.funcignore` excludes local secrets, documents, model weights, outputs, tests and reference archives. Only the three public schema examples are included from `examples`.

Publishing requires Azure CLI, Azure Functions Core Tools v4, and permission to deploy to your Azure subscription. Local setup does not create or publish Azure resources.

## Call the APIs

Retrieve a function/host key from the Function App. Use your app's default hostname shown in Azure Portal (it may include a generated suffix):

```sh
curl -X POST "https://YOUR_APP_HOSTNAME/classify" \
  -H "x-functions-key: YOUR_FUNCTION_KEY" \
  -F 'file=@document.pdf' \
  -F 'schema=<examples/classification.schema.json'

curl -X POST "https://YOUR_APP_HOSTNAME/extract" \
  -H "x-functions-key: YOUR_FUNCTION_KEY" \
  -F 'file=@document.pdf' \
  -F 'schema=<examples/manulife-critical.schema.json' \
  -F 'ocr_pages=<classification-response.json'
```

Routes have no added `/api` prefix. To access Swagger, open `https://YOUR_APP_HOSTNAME/docs?code=YOUR_FUNCTION_KEY`; the OpenAPI fetch and API calls also require the key, so use a browser header extension setting `x-functions-key` on your app's hostname. Avoid sharing URLs containing keys. The localhost FastAPI server does not require a Functions key.

For a website, call these APIs through your website's backend and keep the function key there. If direct browser calls are required, configure allowed origins in the Function App's CORS settings and use per-user authentication rather than embedding a shared function key in public JavaScript.

## HTTP duration limit

Azure HTTP requests have a **230-second response limit**. This deployment uses a **210-second inference worker timeout** and a 220-second function timeout so ordinary overruns can return an error. Existing extraction quality checks and retry counts are unchanged, but longer documents or multiple crop retries can exceed this budget. This synchronous deployment does not guarantee completion for every 20-page PDF. Supporting longer jobs requires a queue/background worker plus polling for results; increasing `functionTimeout` alone does not remove the HTTP limit.

References: [Azure FastAPI sample](https://github.com/Azure-Samples/fastapi-on-azure-functions), [Python remote build](https://learn.microsoft.com/en-us/azure/azure-functions/python-build-options), [Flex Consumption setup](https://learn.microsoft.com/en-us/azure/azure-functions/flex-consumption-how-to), [HTTP timeout limits](https://learn.microsoft.com/en-us/azure/azure-functions/functions-scale).
