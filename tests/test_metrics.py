from actserve.metrics import SchedulerMetrics, prometheus_text


def test_prometheus_export_contains_core_metrics() -> None:
    metrics = SchedulerMetrics()
    metrics.record_submit()
    text = prometheus_text(metrics.snapshot())
    assert "actserve_submitted 1" in text
    assert "actserve_deadline_miss_rate 0.0" in text
