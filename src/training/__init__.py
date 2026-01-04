"""Training modules for learnable VST."""

from .losses import (
    HomoscedasticityLoss,
    VarianceFlatnessLoss,
    ShapePenalty,
    TransformRegularization,
    CombinedTransformLoss,
    DenoiserLoss,
)

from .gauge_fixing import (
    RunningStats,
    GaugeFixingManager,
    compute_standardization_stats,
    check_gauge_quality,
)

from .diagnostics import (
    DiagnosticResult,
    ConvergenceDiagnostics,
    GaugeQualityMonitor,
    BlindSpotLeakageDetector,
    TransformQualityMonitor,
    ResidualStatisticsMonitor,
    DiagnosticSuite,
)

from .alternating_trainer import (
    TrainerConfig,
    AlternatingTrainer,
    LightweightTrainer,
)

__all__ = [
    # Losses
    "HomoscedasticityLoss",
    "VarianceFlatnessLoss",
    "ShapePenalty",
    "TransformRegularization",
    "CombinedTransformLoss",
    "DenoiserLoss",
    # Gauge fixing
    "RunningStats",
    "GaugeFixingManager",
    "compute_standardization_stats",
    "check_gauge_quality",
    # Diagnostics
    "DiagnosticResult",
    "ConvergenceDiagnostics",
    "GaugeQualityMonitor",
    "BlindSpotLeakageDetector",
    "TransformQualityMonitor",
    "ResidualStatisticsMonitor",
    "DiagnosticSuite",
    # Trainer
    "TrainerConfig",
    "AlternatingTrainer",
    "LightweightTrainer",
]
