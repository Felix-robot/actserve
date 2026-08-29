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

## Run as a standalone control plane

The server command starts ActServe in front of the policy endpoint and listens
on localhost by default:

```bash
pip install "actserve[server]"
actserve serve --backend-url http://127.0.0.1:9000/infer
```

From a source checkout, the complete two-process smoke test is:

```bash
# terminal 1: example policy service
uv run uvicorn examples.http_policy_server:app --port 9000

# terminal 2: ActServe control plane
uv run actserve serve --backend-url http://127.0.0.1:9000/infer

# terminal 3: one observation
curl -s http://127.0.0.1:8080/v1/actions \
  -H 'content-type: application/json' \
  -d '{"session_id":"robot-1","model":"example","sequence_no":1,"deadline_ms":100,"observation":{"frame":1}}'
```

The example endpoint only echoes observations and is never a valid robot
policy.

To require authentication on ActServe and authenticate to the backend, put the
tokens in environment variables and pass only their names:

```bash
export ACTSERVE_API_KEY='...'
export POLICY_API_KEY='...'
actserve serve \
  --backend-url https://policy.example/infer \
  --api-key-env ACTSERVE_API_KEY \
  --backend-token-env POLICY_API_KEY
```

The service exposes unauthenticated `/healthz` and `/readyz` probes. When an API
key is configured, `/v1/actions`, `/v1/metrics`, and `/metrics` require a Bearer
token. Binding to a non-loopback interface remains an explicit operator choice.

The standalone command bounds the pending queue at 1024 requests by default.
Tune it with `--max-pending-requests`; when a new session would exceed the
limit, ActServe returns HTTP `429` with `Retry-After: 1`. A newer observation
may still replace an already queued observation from the same session, so
backpressure does not prevent latest-frame coalescing.

The HTTP backend learns a rolling p90 end-to-end latency independently for each
observed batch size. Estimates include a configurable safety factor and use a
conservative scaled estimate for unseen larger batches. To enable predictive
deadline rejection from startup:

```bash
actserve serve \
  --backend-url http://127.0.0.1:9000/infer \
  --initial-backend-latency-ms 80 \
  --latency-safety-factor 1.15 \
  --drop-unserviceable-requests
```

The initial value should come from an isolated warmup on the same model and
hardware. Predictive rejection is opt-in because a poor estimate can discard
requests that might otherwise complete.
