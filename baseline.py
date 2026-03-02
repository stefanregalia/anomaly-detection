#!/usr/bin/env python3
import json
import math
import logging
import boto3
from datetime import datetime
from typing import Optional

# Logging setup

# Reuse the same logger configured in app.py
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")

class BaselineManager:
    """
    Maintains a per-channel running baseline using Welford's online algorithm,
    which computes mean and variance incrementally without storing all past data.
    """
    def __init__(self, bucket: str, baseline_key: str = "state/baseline.json"):
        self.bucket = bucket
        self.baseline_key = baseline_key

    def load(self) -> dict:
        # Returns an empty baseline if the file doesn't exist yet
        try:
            response = s3.get_object(Bucket=self.bucket, Key=self.baseline_key)
            baseline = json.loads(response["Body"].read())
            logger.info(f"Loaded baseline from s3://{self.bucket}/{self.baseline_key}")
            return baseline
        except s3.exceptions.NoSuchKey:
            logger.info("No existing baseline found, starting fresh")
            return {}
        except Exception as e:
            logger.exception("Failed to load baseline from S3")
            return {}

    def save(self, baseline: dict):
        try:
            baseline["last_updated"] = datetime.utcnow().isoformat()
            s3.put_object(
                Bucket=self.bucket,
                Key=self.baseline_key,
                Body=json.dumps(baseline, indent=2),
                ContentType="application/json"
            )
            logger.info(f"Baseline saved to s3://{self.bucket}/{self.baseline_key}")

            # Sync application log to S3 whenever baseline is saved
            log_path = "/opt/anomaly-detection/app.log"
            s3.upload_file(
                log_path,
                self.bucket,
                "logs/app.log"
            )
            logger.info(f"Application log synced to s3://{self.bucket}/logs/app.log")

        except Exception as e:
            logger.exception("Failed to save baseline or sync logs to S3")

    def update(self, baseline: dict, channel: str, new_values: list[float]) -> dict:
        """
        Welford's online algorithm for numerically stable mean and variance.
        Each channel tracks: count, mean, M2 (sum of squared deviations).
        Variance = M2 / count, std = sqrt(variance).
        """
        if channel not in baseline:
            baseline[channel] = {"count": 0, "mean": 0.0, "M2": 0.0}

        state = baseline[channel]

        for value in new_values:
            state["count"] += 1
            delta = value - state["mean"]
            state["mean"] += delta / state["count"]
            delta2 = value - state["mean"]
            state["M2"] += delta * delta2

        # Only compute std once we have enough observations
        if state["count"] >= 2:
            variance = state["M2"] / state["count"]
            state["std"] = math.sqrt(variance)
        else:
            state["std"] = 0.0

        logger.info(
            f"Channel '{channel}' baseline updated: "
            f"count={state['count']}, mean={round(state['mean'], 4)}, std={round(state['std'], 4)}"
        )

        baseline[channel] = state
        return baseline

    def get_stats(self, baseline: dict, channel: str) -> Optional[dict]:
        return baseline.get(channel)