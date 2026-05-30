"""Fraud detection thesis — modular Python package.

Quick imports:
    from src import FraudDataset, FraudScorer, FraudTrainer
    from src import DataQualityReport, ModelRegistry
    from src import charts
"""
from .features import FraudDataset, build_feature_frame
from .quality import DataQualityReport
from .models import ModelRegistry
from .evaluation import EvaluationResult
from .training import FraudTrainer
from .scoring import FraudScorer
from .io import load_csv, load_artifact, save_artifact, ArtifactBundle

__all__ = [
    "FraudDataset",
    "build_feature_frame",
    "DataQualityReport",
    "ModelRegistry",
    "EvaluationResult",
    "FraudTrainer",
    "FraudScorer",
    "load_csv",
    "load_artifact",
    "save_artifact",
    "ArtifactBundle",
]
