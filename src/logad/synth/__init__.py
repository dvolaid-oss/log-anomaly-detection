"""Synthetic data generators (safe to publish, exact ground truth)."""

from logad.synth import auth_log

GENERATORS = {"auth_log": auth_log.generate}
