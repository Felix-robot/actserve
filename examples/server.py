"""Development-only HTTP server using the simulated public backend.

Run with:
    uv sync --extra server
    uv run uvicorn examples.server:app
"""

from actserve.backend import SimulatedBackend
from actserve.scheduler import Scheduler, SchedulerConfig
from actserve.server import create_app

backend = SimulatedBackend(fixed_ms=20, per_item_ms=2, max_batch_size=8)
scheduler = Scheduler(backend, SchedulerConfig())
app = create_app(scheduler)
