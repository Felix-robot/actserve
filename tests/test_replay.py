from actserve.replay import outcome_record
from actserve.types import InferenceRequest, RequestOutcome, ResultStatus


def test_public_record_omits_observation_and_action() -> None:
    request = InferenceRequest.with_timeout(
        session_id="robot",
        model="model",
        observation={"private_pixels": [1, 2, 3]},
        timeout_ms=100,
        sequence_no=4,
    )
    outcome = RequestOutcome(
        request=request,
        status=ResultStatus.COMPLETED,
        action=None,
        dispatched_ns=request.received_ns,
        completed_ns=request.received_ns + 1,
    )
    record = outcome_record(outcome)
    assert "observation" not in record
    assert "action" not in record
    assert record["schema"] == "actserve.outcome.v1"
