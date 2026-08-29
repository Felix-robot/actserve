# JSON/HTTP backend

`HttpJsonBackend` lets ActServe schedule an existing policy service without
embedding that model's Python environment into ActServe. Install the optional
client dependency:

```bash
pip install "actserve[http]"
```

The backend sends `POST` requests with this shape:

```json
{
  "requests": [
    {
      "request_id": "opaque-id",
      "session_id": "robot-1",
      "model": "my-policy",
      "sequence_no": 42,
      "observation": {"state": [0.1, 0.2]},
      "metadata": {}
    }
  ]
}
```

The policy service returns one identity-preserving result per input, in the
same order:

```json
{
  "actions": [
    {
      "request_id": "opaque-id",
      "session_id": "robot-1",
      "model": "my-policy",
      "sequence_no": 42,
      "actions": [[0.0, 0.1]],
      "metadata": {"server_total_ms": 18.4}
    }
  ]
}
```

ActServe rejects missing, extra, malformed, or identity-mismatched results. A
backend HTTP error becomes an explicit failed request outcome. Observations and
actions are forwarded in memory and are not added to ActServe traces.

```python
from actserve.integrations.http_json import HttpJsonBackend

backend = HttpJsonBackend(
    "http://127.0.0.1:9000/infer",
    max_batch_size=8,
    timeout_ms=30_000,
)
```

Requests are batched only when endpoint, logical model, and optional
`metadata.input_signature` match. Use TLS and authentication for any non-local
backend; never place tokens directly in committed source or shell history.
