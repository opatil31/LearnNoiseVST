"""
Gaussian copula for modeling dependence structure.

When noise components across groups (channels or features) are correlated,
we model the dependence structure using a Gaussian copula.

A copula separates the marginal distributions from the dependence structure:
    F(u_1, ..., u_d) = C(F_1(u_1), ..., F_d(u_d))

The Gaussian copula:
1. Transforms each marginal to uniform via probability integral transform
2. Converts uniforms to standard normals via Φ^{-1}
3. Models the resulting multivariate normal with correlation matrix R
4. For sampling: generate correlated normals → convert to uniforms → apply marginal quantiles

This allows flexible marginals with Gaussian-like dependence structure.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class CopulaFitResult:
    """Result of Gaussian copula fitting."""
    correlation_matrix: np.ndarray
    shrinkage_applied: float
    eigenvalues: np.ndarray
    condition_number: float
    is_well_conditioned: bool


class GaussianCopula:
    """
    Gaussian copula for modeling dependence between groups.

    For image channels or correlated tabular features.

    The copula is fitted by:
    1. Converting each group's residuals to normal scores via their marginal CDFs
    2. Estimating the correlation matrix of the normal scores
    3. Optionally applying shrinkage for numerical stability

    Args:
        shrinkage: Shrinkage intensity towards identity matrix (0 = no shrinkage).
            Recommended for small sample sizes or high dimensions.
        min_eigenvalue: Minimum eigenvalue to ensure positive definiteness.
    """

    def __init__(
        self,
        shrinkage: float = 0.1,
        min_eigenvalue: float = 1e-4,
    ):
        self.shrinkage = shrinkage
        self.min_eigenvalue = min_eigenvalue

        # Fitted parameters
        self.correlation: Optional[np.ndarray] = None
        self.cholesky: Optional[np.ndarray] = None
        self.dimension: int = 0
        self.group_ids: List[int] = []

    def fit(
        self,
        u_matrix: np.ndarray,
        marginals: Optional['MarginalCollection'] = None,
        group_ids: Optional[List[int]] = None,
    ) -> CopulaFitResult:
        """
        Fit correlation from residuals.

        Args:
            u_matrix: [N, d] matrix of residuals, each column is a group.
            marginals: MarginalCollection for converting to normal scores.
                If None, assumes residuals are already approximately normal.
            group_ids: List of group IDs corresponding to columns.

        Returns:
            CopulaFitResult with correlation matrix and diagnostics.
        """
        u_matrix = np.asarray(u_matrix)
        N, d = u_matrix.shape

        if N < d + 1:
            raise ValueError(f"Need at least {d+1} samples for {d} dimensions, got {N}")

        self.dimension = d
        self.group_ids = group_ids if group_ids is not None else list(range(d))

        # Convert to normal scores via probability integral transform
        if marginals is not None:
            y = self._to_normal_scores(u_matrix, marginals)
        else:
            # Assume already standardized; use empirical normal transformation
            y = self._empirical_normal_transform(u_matrix)

        # Estimate correlation with shrinkage
        self.correlation = self._estimate_correlation(y)

        # Ensure positive definiteness and compute Cholesky
        self.cholesky = self._compute_cholesky(self.correlation)

        # Diagnostics
        eigenvalues = np.linalg.eigvalsh(self.correlation)
        condition_number = eigenvalues.max() / (eigenvalues.min() + 1e-10)

        return CopulaFitResult(
            correlation_matrix=self.correlation.copy(),
            shrinkage_applied=self.shrinkage,
            eigenvalues=eigenvalues,
            condition_number=condition_number,
            is_well_conditioned=condition_number < 1e6,
        )

    def _to_normal_scores(
        self,
        u_matrix: np.ndarray,
        marginals: 'MarginalCollection',
    ) -> np.ndarray:
        """Convert to normal scores via probability integral transform."""
        try:
            from scipy.stats import norm
            use_scipy = True
        except ImportError:
            use_scipy = False

        N, d = u_matrix.shape
        y = np.zeros_like(u_matrix)

        for j, group_id in enumerate(self.group_ids):
            if group_id in marginals:
                # Get CDF values
                p = marginals.cdf(group_id, u_matrix[:, j])
            else:
                # Fallback: empirical CDF
                p = self._empirical_cdf(u_matrix[:, j])

            # Clip to avoid infinities
            p = np.clip(p, 1e-6, 1 - 1e-6)

            # Convert to normal scores
            if use_scipy:
                y[:, j] = norm.ppf(p)
            else:
                # Approximate inverse normal CDF
                y[:, j] = self._approximate_normal_ppf(p)

        return y

    def _empirical_normal_transform(self, u_matrix: np.ndarray) -> np.ndarray:
        """Transform each column to normal scores using rank transformation."""
        try:
            from scipy.stats import norm
            use_scipy = True
        except ImportError:
            use_scipy = False

        N, d = u_matrix.shape
        y = np.zeros_like(u_matrix)

        for j in range(d):
            # Rank-based transformation
            ranks = np.argsort(np.argsort(u_matrix[:, j]))
            p = (ranks + 0.5) / N

            if use_scipy:
                y[:, j] = norm.ppf(p)
            else:
                y[:, j] = self._approximate_normal_ppf(p)

        return y

    def _empirical_cdf(self, x: np.ndarray) -> np.ndarray:
        """Compute empirical CDF via ranks."""
        ranks = np.argsort(np.argsort(x))
        return (ranks + 0.5) / len(x)

    def _approximate_normal_ppf(self, p: np.ndarray) -> np.ndarray:
        """Approximate standard normal quantile function."""
        # Rational approximation (Abramowitz and Stegun)
        p = np.clip(p, 1e-6, 1 - 1e-6)

        sign = np.where(p < 0.5, -1, 1)
        p_adj = np.where(p < 0.5, p, 1 - p)

        t = np.sqrt(-2 * np.log(p_adj))

        # Coefficients for approximation
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        x = t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)

        return sign * x

    def _estimate_correlation(self, y: np.ndarray) -> np.ndarray:
        """Estimate correlation matrix with shrinkage."""
        N, d = y.shape

        # Sample correlation
        corr_raw = np.corrcoef(y, rowvar=False)

        # Handle NaN from constant columns
        corr_raw = np.nan_to_num(corr_raw, nan=0.0)

        # Shrink towards identity
        corr = (1 - self.shrinkage) * corr_raw + self.shrinkage * np.eye(d)

        return corr

    def _compute_cholesky(self, corr: np.ndarray) -> np.ndarray:
        """Compute Cholesky decomposition, fixing eigenvalues if needed."""
        d = corr.shape[0]

        # Check eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(corr)

        # Fix any non-positive eigenvalues
        min_eig = eigenvalues.min()
        if min_eig < self.min_eigenvalue:
            # Adjust eigenvalues
            eigenvalues = np.maximum(eigenvalues, self.min_eigenvalue)

            # Reconstruct matrix
            corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

            # Re-normalize diagonal to 1
            d_inv = 1.0 / np.sqrt(np.diag(corr))
            corr = corr * np.outer(d_inv, d_inv)

        try:
            chol = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            # Last resort: use eigendecomposition
            eigenvalues = np.maximum(eigenvalues, self.min_eigenvalue)
            sqrt_eig = np.sqrt(eigenvalues)
            chol = eigenvectors @ np.diag(sqrt_eig)

        return chol

    def sample(
        self,
        n: int,
        marginals: Optional['MarginalCollection'] = None,
    ) -> np.ndarray:
        """
        Sample n vectors with fitted correlation structure.

        Args:
            n: Number of samples.
            marginals: MarginalCollection for applying marginal quantiles.
                If None, returns correlated normal samples.

        Returns:
            [n, d] array of samples.
        """
        if self.cholesky is None:
            raise ValueError("Copula not fitted. Call fit() first.")

        # Sample independent standard normals
        z = np.random.randn(n, self.dimension)

        # Apply correlation via Cholesky
        y = z @ self.cholesky.T

        if marginals is None:
            return y

        # Convert to uniforms via Gaussian CDF
        try:
            from scipy.stats import norm
            p = norm.cdf(y)
        except ImportError:
            p = 0.5 * (1 + np.tanh(y / np.sqrt(2)))  # Approximation

        # Apply inverse marginal CDFs
        u = np.zeros_like(p)
        for j, group_id in enumerate(self.group_ids):
            if group_id in marginals:
                u[:, j] = marginals.sample_from_uniform(group_id, p[:, j])
            else:
                # Fallback: return normal scores
                u[:, j] = y[:, j]

        return u

    def sample_from_uniforms(
        self,
        p_independent: np.ndarray,
        marginals: 'MarginalCollection',
    ) -> np.ndarray:
        """
        Apply copula transformation to independent uniforms.

        Takes independent U[0,1] variables and returns correlated samples
        with the fitted dependence structure.

        Args:
            p_independent: [n, d] independent uniforms.
            marginals: MarginalCollection for marginal transforms.

        Returns:
            [n, d] correlated samples.
        """
        if self.cholesky is None:
            raise ValueError("Copula not fitted. Call fit() first.")

        try:
            from scipy.stats import norm
            use_scipy = True
        except ImportError:
            use_scipy = False

        # Convert independent uniforms to independent normals
        if use_scipy:
            z = norm.ppf(p_independent)
        else:
            z = self._approximate_normal_ppf(p_independent)

        # Apply correlation
        y = z @ self.cholesky.T

        # Convert back to uniforms
        if use_scipy:
            p_correlated = norm.cdf(y)
        else:
            p_correlated = 0.5 * (1 + np.tanh(y / np.sqrt(2)))

        # Apply marginal inverse CDFs
        u = np.zeros_like(p_correlated)
        for j, group_id in enumerate(self.group_ids):
            u[:, j] = marginals.sample_from_uniform(group_id, p_correlated[:, j])

        return u

    def get_correlation_matrix(self) -> np.ndarray:
        """Return the fitted correlation matrix."""
        if self.correlation is None:
            raise ValueError("Copula not fitted. Call fit() first.")
        return self.correlation.copy()


class IndependenceCopula:
    """
    Independence copula (no dependence between groups).

    Useful as a baseline or when groups are approximately independent.
    """

    def __init__(self):
        self.dimension = 0
        self.group_ids: List[int] = []

    def fit(
        self,
        u_matrix: np.ndarray,
        marginals: Optional['MarginalCollection'] = None,
        group_ids: Optional[List[int]] = None,
    ) -> CopulaFitResult:
        """Fit (no-op for independence)."""
        N, d = u_matrix.shape
        self.dimension = d
        self.group_ids = group_ids if group_ids is not None else list(range(d))

        return CopulaFitResult(
            correlation_matrix=np.eye(d),
            shrinkage_applied=0.0,
            eigenvalues=np.ones(d),
            condition_number=1.0,
            is_well_conditioned=True,
        )

    def sample(
        self,
        n: int,
        marginals: Optional['MarginalCollection'] = None,
    ) -> np.ndarray:
        """Sample independently from each marginal."""
        if marginals is None:
            return np.random.randn(n, self.dimension)

        u = np.zeros((n, self.dimension))
        for j, group_id in enumerate(self.group_ids):
            u[:, j] = marginals.sample(group_id, n)

        return u


def check_independence(
    u_matrix: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[bool, Dict]:
    """
    Check if groups are approximately independent.

    Uses pairwise correlation tests.

    Args:
        u_matrix: [N, d] residual matrix.
        alpha: Significance level.

    Returns:
        (is_independent, details_dict)
    """
    N, d = u_matrix.shape

    # Compute all pairwise correlations
    corr_matrix = np.corrcoef(u_matrix, rowvar=False)

    # Test each off-diagonal element
    significant_correlations = []

    for i in range(d):
        for j in range(i + 1, d):
            r = corr_matrix[i, j]

            # Fisher transformation
            z = 0.5 * np.log((1 + r) / (1 - r + 1e-8))
            se = 1 / np.sqrt(N - 3)

            # Two-sided test
            z_stat = abs(z) / se
            p_value = 2 * (1 - 0.5 * (1 + np.tanh(z_stat / np.sqrt(2))))  # Approx normal CDF

            if p_value < alpha:
                significant_correlations.append({
                    'i': i,
                    'j': j,
                    'correlation': r,
                    'p_value': p_value,
                })

    is_independent = len(significant_correlations) == 0

    details = {
        'correlation_matrix': corr_matrix,
        'significant_pairs': significant_correlations,
        'max_abs_correlation': np.max(np.abs(corr_matrix - np.eye(d))),
    }

    return is_independent, details


def choose_copula(
    u_matrix: np.ndarray,
    marginals: Optional['MarginalCollection'] = None,
    independence_threshold: float = 0.1,
    shrinkage: float = 0.1,
) -> Union[GaussianCopula, IndependenceCopula]:
    """
    Automatically choose between independence and Gaussian copula.

    Args:
        u_matrix: [N, d] residual matrix.
        marginals: Marginal distributions.
        independence_threshold: Maximum correlation for independence.
        shrinkage: Shrinkage for Gaussian copula.

    Returns:
        Fitted copula (either independence or Gaussian).
    """
    is_independent, details = check_independence(u_matrix)

    max_corr = details['max_abs_correlation']

    if is_independent or max_corr < independence_threshold:
        copula = IndependenceCopula()
    else:
        copula = GaussianCopula(shrinkage=shrinkage)

    copula.fit(u_matrix, marginals)

    return copula
