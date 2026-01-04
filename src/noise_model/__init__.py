"""Noise model for characterizing and sampling noise."""

from .calibration import (
    ResidualData,
    CalibrationResult,
    CalibrationDatasetGenerator,
    ResidualAnalyzer,
)

from .location_scale import (
    LocationScaleFitResult,
    LocationScaleModel,
    LocationScaleModelCollection,
    DeltaMethodVariancePropagator,
)

from .marginal_fit import (
    MarginalFitResult,
    EmpiricalMarginal,
    GaussianMarginal,
    MarginalCollection,
)

from .copula import (
    CopulaFitResult,
    GaussianCopula,
    IndependenceCopula,
    check_independence,
    choose_copula,
)

from .sampler import (
    NoiseModelConfig,
    NoiseModelFitResult,
    NoiseModelSampler,
    NoiseAugmenter,
    NoiseModelIO,
    fit_noise_model,
    validate_noise_model,
)

__all__ = [
    # Calibration
    "ResidualData",
    "CalibrationResult",
    "CalibrationDatasetGenerator",
    "ResidualAnalyzer",
    # Location-scale
    "LocationScaleFitResult",
    "LocationScaleModel",
    "LocationScaleModelCollection",
    "DeltaMethodVariancePropagator",
    # Marginal
    "MarginalFitResult",
    "EmpiricalMarginal",
    "GaussianMarginal",
    "MarginalCollection",
    # Copula
    "CopulaFitResult",
    "GaussianCopula",
    "IndependenceCopula",
    "check_independence",
    "choose_copula",
    # Sampler
    "NoiseModelConfig",
    "NoiseModelFitResult",
    "NoiseModelSampler",
    "NoiseAugmenter",
    "NoiseModelIO",
    "fit_noise_model",
    "validate_noise_model",
]
