from detectors_bench.metrics import binary_metrics
from detectors_bench.schemas import Prediction


def test_binary_metrics_low_fpr_keys():
    preds = [
        Prediction(id="h1", detector="x", label=0, score_ai=0.1),
        Prediction(id="h2", detector="x", label=0, score_ai=0.2),
        Prediction(id="a1", detector="x", label=1, score_ai=0.8),
        Prediction(id="a2", detector="x", label=1, score_ai=0.9),
    ]
    metrics = binary_metrics(preds)
    assert metrics["auroc"] == 1.0
    assert metrics["tpr_at_fpr_0.01"] == 1.0
    assert metrics["f1"] == 1.0
