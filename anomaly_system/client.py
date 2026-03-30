from __future__ import annotations

import argparse
import time

import httpx

from anomaly_system.data import make_synthetic_sensor_data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Predict endpoint URL")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--n-features", type=int, default=16)
    p.add_argument("--anomaly-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    X, y = make_synthetic_sensor_data(
        n=args.n,
        n_features=args.n_features,
        anomaly_fraction=args.anomaly_fraction,
        seed=args.seed,
    )

    with httpx.Client(timeout=10.0) as client:
        for i in range(args.n):
            payload = {
                "event_id": f"evt-{i}",
                "ts": time.time(),
                "values": X[i].tolist(),
            }
            r = client.post(args.url, json=payload)
            r.raise_for_status()
            out = r.json()
            print(
                f"{out['event_id']} injected={bool(y[i])} pred={out['is_anomaly']} "
                f"score={out['score']:.6f} th={out['threshold']:.6f} latency_ms={out['latency_ms']:.2f}"
            )


if __name__ == "__main__":
    main()

