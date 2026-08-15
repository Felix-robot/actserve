"""Template for wrapping an existing synchronous policy client privately."""

import asyncio

from actserve.backend import CallableBackend
from actserve.types import ActionChunk


class ExistingPolicyClient:
    def infer(self, observation):
        """Replace this body with an existing WebSocket/RPC policy call."""
        return {"actions": observation}


client = ExistingPolicyClient()


async def infer_batch(requests):
    if len(requests) != 1:
        raise ValueError("this existing policy client only supports batch size 1")
    request = requests[0]
    actions = await asyncio.to_thread(client.infer, request.observation)
    return [
        ActionChunk(
            request_id=request.request_id,
            session_id=request.session_id,
            sequence_no=request.sequence_no,
            actions=actions,
            model=request.model,
        )
    ]


backend = CallableBackend(infer_batch, max_batch_size=1)
