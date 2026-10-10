"""Readable summary of the service's /metrics (no Prometheus or Grafana needed).

    python -m serving.metrics_summary [--url http://localhost:8000]
"""
import argparse
import re
import urllib.request

_SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)$')
_LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def parse(text: str) -> list:
    """[(name, {label: value}, float)] from Prometheus text."""
    samples = []
    for line in text.splitlines():
        m = _SAMPLE.match(line.strip())
        if line.startswith("#") or not m:
            continue
        labels = {k: v.replace('\\"', '"').replace("\\\\", "\\") for k, v in _LABEL.findall(m.group(2) or "")}
        samples.append((m.group(1), labels, float(m.group(3))))
    return samples


def _histogram(samples, name):
    buckets = sorted(((float(l["le"]), v) for n, l, v in samples if n == f"{name}_bucket"
                      for l in [l] if l["le"] != "+Inf"))
    count = next((v for n, _, v in samples if n == f"{name}_count"), 0.0)
    total = next((v for n, _, v in samples if n == f"{name}_sum"), 0.0)
    return buckets, count, total


def quantile(buckets, count, q):
    """Estimate a quantile from cumulative histogram buckets by linear interpolation."""
    if not count:
        return None
    target, previous_bound, previous_n = q * count, 0.0, 0.0
    for bound, n in buckets:
        if n >= target:
            span = n - previous_n
            return bound if span == 0 else previous_bound + (bound - previous_bound) * (target - previous_n) / span
        previous_bound, previous_n = bound, n
    return buckets[-1][0] if buckets else None


def summarize(text: str) -> str:
    s = parse(text)
    predict_requests = sum(v for n, l, v in s if n == "plant_requests_total" and l["endpoint"] == "/predict")
    successes = sum(v for n, _, v in s if n == "plant_predictions_total")
    rejected = {l["reason"]: v for n, l, v in s if n == "plant_rejected_requests_total"}
    errors = {l["category"]: v for n, l, v in s if n == "plant_errors_total"}
    model = next((l for n, l, _ in s if n == "plant_model_info"), None)
    lines = [f"Model served: {model['name']} @{model['alias']} version {model['version']}" if model
             else "Model served: none loaded"]
    lines.append(f"/predict requests: {predict_requests:.0f}   successful: {successes:.0f}   "
                 f"rejected: {sum(rejected.values()):.0f}   server errors: {sum(errors.values()):.0f}")
    if predict_requests:
        lines.append(f"Rejection rate: {sum(rejected.values()) / predict_requests:.1%}")
    for title, items in (("Rejections", rejected), ("Errors", errors)):
        if items:
            lines.append(f"{title} by reason: " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(items.items())))
    for title, name in (("Predict latency", "plant_predict_duration_seconds"),
                        ("Inference latency", "plant_inference_duration_seconds")):
        buckets, count, _ = _histogram(s, name)
        if count:
            lines.append(f"{title}: p50 ~ {quantile(buckets, count, 0.5) * 1000:.0f} ms, "
                         f"p95 ~ {quantile(buckets, count, 0.95) * 1000:.0f} ms (bucket estimates)")
    buckets, count, total = _histogram(s, "plant_prediction_confidence")
    if count:
        below = next((n for bound, n in buckets if bound >= 0.7), 0.0)
        lines.append(f"Confidence: mean {total / count:.3f}; {below / count:.1%} of predictions below 0.7")
    classes = sorted(((l["label"], v) for n, l, v in s if n == "plant_predictions_total"), key=lambda kv: -kv[1])
    if classes:
        lines.append("Predicted classes: " + ", ".join(f"{k} {v:.0f}" for k, v in classes))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarise the service's /metrics.")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args(argv)
    with urllib.request.urlopen(args.url.rstrip("/") + "/metrics", timeout=5) as response:
        print(summarize(response.read().decode()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
