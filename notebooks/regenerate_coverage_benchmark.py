"""
Regenerate the coverage-utility benchmark and its analysis tables by running the
relevant cells of notebook 02 in order (Steps 124 to 133), without the other
experiments of the notebook.

Usage:
    python notebooks/setup_workspace.py <workspace>      # creates the folder layout
    MST_PROJECT_ROOT=<workspace> python notebooks/regenerate_coverage_benchmark.py

Outputs go to <workspace>/05_Results and <workspace>/07_Tables. The main file,
coverage_exact_vs_greedy_main_benchmark_v1_pool_normalized.csv, is stored
compressed in data/coverage/. Running time is about 30 minutes.
"""

# ===== notebook 02, code cell 0 =====
print('### notebook cell 0', flush=True)
import os
PROJECT_ROOT = os.environ.get("MST_PROJECT_ROOT", "..")
from pathlib import Path
import sys
import random

import numpy as np
import pandas as pd


# --------------------------------------------------
# Project directories
# --------------------------------------------------

BASE_DIR = Path(
    PROJECT_ROOT
)

PYTHON_DIR = (
    BASE_DIR
    / "03_Python_Optimization"
)

RESULTS_DIR = (
    BASE_DIR
    / "05_Results"
)

FIGURES_DIR = (
    BASE_DIR
    / "06_Figures"
)

TABLES_DIR = (
    BASE_DIR
    / "07_Tables"
)


# --------------------------------------------------
# Make local MST module importable
# --------------------------------------------------

python_path = str(PYTHON_DIR)

if python_path not in sys.path:
    sys.path.insert(
        0,
        python_path,
    )

import mst_core


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

GLOBAL_SEED = 2026

random.seed(
    GLOBAL_SEED
)

np.random.seed(
    GLOBAL_SEED
)


print("MST experimental environment initialized.")
print("Random seed:", GLOBAL_SEED)
print("MST module:", mst_core.__file__)
print("Results folder:", RESULTS_DIR)
print("Figures folder:", FIGURES_DIR)
print("Tables folder:", TABLES_DIR)

# ===== notebook 02, code cell 1 =====
print('### notebook cell 1', flush=True)
# --------------------------------------------------
# Computational experiment design
# --------------------------------------------------

# Candidate-set sizes where Exact-MST remains practical.
EXACT_BENCHMARK_SIZES = [
    8,
    10,
    12,
    14,
    16,
]

# Larger candidate sets used later for scalability analysis.
SCALABILITY_SIZES = [
    16,
    32,
    64,
    128,
    256,
]

# Sufficiency requirements spanning moderate to stringent conditions.
TAU_LEVELS = [
    0.70,
    0.80,
    0.90,
    0.95,
]

# Objective-weight configurations.
#
# Each configuration sums to 1 so that the relative influence
# of privacy, interpretation cost, and parsimony is transparent.
WEIGHT_PROFILES = {
    "balanced": {
        "lambda_privacy": 1 / 3,
        "lambda_cognitive": 1 / 3,
        "lambda_cardinality": 1 / 3,
    },

    "privacy_priority": {
        "lambda_privacy": 0.60,
        "lambda_cognitive": 0.20,
        "lambda_cardinality": 0.20,
    },

    "interpretation_priority": {
        "lambda_privacy": 0.20,
        "lambda_cognitive": 0.60,
        "lambda_cardinality": 0.20,
    },

    "parsimony_priority": {
        "lambda_privacy": 0.20,
        "lambda_cognitive": 0.20,
        "lambda_cardinality": 0.60,
    },
}

# Structural regimes controlling the relationship between
# information utility and disclosure burden.
INSTANCE_REGIMES = [
    "independent",
    "utility_burden_conflict",
    "utility_burden_alignment",
]

print("Exact benchmark sizes:", EXACT_BENCHMARK_SIZES)
print("Scalability sizes:", SCALABILITY_SIZES)
print("Tau levels:", TAU_LEVELS)
print("Weight profiles:", list(WEIGHT_PROFILES))
print("Instance regimes:", INSTANCE_REGIMES)

# ===== notebook 02, code cell 2 =====
print('### notebook cell 2', flush=True)
# --------------------------------------------------
# Synthetic MST benchmark-instance generator
# --------------------------------------------------

REGIME_STRENGTH = 0.75


def generate_mst_instance(
    n_items: int,
    regime: str,
    seed: int,
    regime_strength: float = REGIME_STRENGTH,
) -> list[mst_core.InformationItem]:
    """
    Generate one reproducible synthetic MST benchmark instance.

    Parameters
    ----------
    n_items
        Number of privacy-admissible candidate information items.

    regime
        Structural relationship between modeled utility and burden:

        "independent"
            Utility, privacy exposure, and interpretation cost
            are generated independently.

        "utility_burden_conflict"
            Higher-utility items tend to have higher privacy
            exposure and interpretation cost.

        "utility_burden_alignment"
            Higher-utility items tend to have lower privacy
            exposure and interpretation cost.

    seed
        Random seed for reproducibility.

    regime_strength
        Strength of the structural relationship in [0, 1].
        A value of 0 gives independence, while larger values
        produce stronger alignment or conflict.

    Returns
    -------
    list[mst_core.InformationItem]
        Generated candidate information items.
    """

    if n_items <= 0:
        raise ValueError(
            "n_items must be greater than zero."
        )

    if regime not in INSTANCE_REGIMES:
        raise ValueError(
            f"Unknown regime: {regime}"
        )

    if not 0.0 <= regime_strength <= 1.0:
        raise ValueError(
            "regime_strength must be in [0, 1]."
        )

    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # Latent modeled utility
    # --------------------------------------------------
    #
    # Beta(2, 2) provides heterogeneous but non-extreme
    # candidate values.
    #
    # The transformation restricts individual utility
    # contributions to [0.08, 0.60], preventing most
    # single items from trivially satisfying stringent
    # thresholds while retaining feasible combinations.
    # --------------------------------------------------

    utility_latent = rng.beta(
        a=2.0,
        b=2.0,
        size=n_items,
    )

    utility_values = (
        0.08
        + 0.52 * utility_latent
    )


    # --------------------------------------------------
    # Independent latent burden components
    # --------------------------------------------------

    privacy_noise = rng.beta(
        a=2.0,
        b=2.0,
        size=n_items,
    )

    cognitive_noise = rng.beta(
        a=2.0,
        b=2.0,
        size=n_items,
    )


    # --------------------------------------------------
    # Structural regime
    # --------------------------------------------------

    if regime == "independent":

        privacy_latent = privacy_noise

        cognitive_latent = cognitive_noise


    elif regime == "utility_burden_conflict":

        # Higher modeled utility tends to imply
        # greater disclosure burden.

        privacy_latent = (
            regime_strength
            * utility_latent
            +
            (1.0 - regime_strength)
            * privacy_noise
        )

        cognitive_latent = (
            regime_strength
            * utility_latent
            +
            (1.0 - regime_strength)
            * cognitive_noise
        )


    elif regime == "utility_burden_alignment":

        # Higher modeled utility tends to imply
        # lower disclosure burden.

        inverse_utility = (
            1.0 - utility_latent
        )

        privacy_latent = (
            regime_strength
            * inverse_utility
            +
            (1.0 - regime_strength)
            * privacy_noise
        )

        cognitive_latent = (
            regime_strength
            * inverse_utility
            +
            (1.0 - regime_strength)
            * cognitive_noise
        )


    # --------------------------------------------------
    # Convert latent burdens to computational scales
    # --------------------------------------------------
    #
    # Per-item privacy values are intentionally bounded
    # below 0.25 to reduce premature saturation of the
    # cumulative bounded privacy function.
    #
    # Cognitive costs are positive raw quantities that
    # are subsequently normalized by MST relative to
    # the complete admissible candidate set.
    # --------------------------------------------------

    privacy_values = (
        0.01
        + 0.24 * privacy_latent
    )

    cognitive_values = (
        0.10
        + 0.90 * cognitive_latent
    )


    # --------------------------------------------------
    # Construct MST information items
    # --------------------------------------------------

    items = []

    for index in range(n_items):

        items.append(
            mst_core.InformationItem(
                item_id=f"z{index + 1:03d}",
                name=(
                    f"Synthetic information item "
                    f"{index + 1}"
                ),
                utility=float(
                    utility_values[index]
                ),
                privacy=float(
                    privacy_values[index]
                ),
                cognitive_cost=float(
                    cognitive_values[index]
                ),
            )
        )

    return items

# ===== notebook 02, code cell 50 =====
print('### notebook cell 50', flush=True)
# --------------------------------------------------
# Marginal-preserving synthetic MST generator
# --------------------------------------------------

REGIME_STRENGTH = 0.75


# Preserve the previous generator for development
# reproducibility if this cell is run for the first time.
if (
    "generate_mst_instance_legacy"
    not in globals()
):
    generate_mst_instance_legacy = (
        generate_mst_instance
    )


def _assign_burden_preserving_marginal(
    base_values: np.ndarray,
    utility_values: np.ndarray,
    regime: str,
    rng: np.random.Generator,
    regime_strength: float,
) -> np.ndarray:
    """
    Reassign an existing set of burden values while
    preserving its exact marginal distribution.

    The returned array contains exactly the same burden
    values as base_values, but their association with
    utility changes according to the requested regime.

    regime_strength controls a noisy rank-association
    mechanism. It is not assumed to equal the resulting
    Pearson or Spearman correlation exactly.
    """

    base_values = np.asarray(
        base_values,
        dtype=float,
    )

    utility_values = np.asarray(
        utility_values,
        dtype=float,
    )

    if (
        base_values.shape
        != utility_values.shape
    ):
        raise ValueError(
            "base_values and utility_values "
            "must have the same shape."
        )

    number_of_items = len(
        base_values
    )

    if number_of_items <= 1:
        return base_values.copy()


    # ----------------------------------------------
    # Independent regime
    # ----------------------------------------------
    #
    # base_values were generated independently of
    # utility_values, so their original assignment
    # already represents the independent condition.
    # ----------------------------------------------

    if regime == "independent":

        return base_values.copy()


    # ----------------------------------------------
    # Construct standardized utility ranks
    # ----------------------------------------------

    utility_order = np.argsort(
        utility_values,
        kind="mergesort",
    )

    utility_ranks = np.empty(
        number_of_items,
        dtype=float,
    )

    utility_ranks[
        utility_order
    ] = np.arange(
        number_of_items,
        dtype=float,
    )

    utility_rank_std = (
        utility_ranks.std(
            ddof=0
        )
    )

    if utility_rank_std > 0:

        standardized_utility_ranks = (
            utility_ranks
            -
            utility_ranks.mean()
        ) / utility_rank_std

    else:

        standardized_utility_ranks = (
            np.zeros(
                number_of_items,
                dtype=float,
            )
        )


    # ----------------------------------------------
    # Create a noisy ordering score
    # ----------------------------------------------

    independent_noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=number_of_items,
    )


    residual_strength = np.sqrt(
        max(
            0.0,
            1.0
            -
            regime_strength ** 2,
        )
    )


    if (
        regime
        == "utility_burden_conflict"
    ):

        direction = 1.0

    elif (
        regime
        == "utility_burden_alignment"
    ):

        direction = -1.0

    else:

        raise ValueError(
            f"Unknown regime: {regime}"
        )


    ordering_score = (
        direction
        * regime_strength
        * standardized_utility_ranks
        +
        residual_strength
        * independent_noise
    )


    # ----------------------------------------------
    # Rank-assign the unchanged burden values
    # ----------------------------------------------
    #
    # Sorting and reassigning preserves the complete
    # empirical marginal distribution exactly.
    # ----------------------------------------------

    score_order = np.argsort(
        ordering_score,
        kind="mergesort",
    )

    sorted_burden_values = np.sort(
        base_values
    )

    assigned_values = np.empty_like(
        base_values,
        dtype=float,
    )

    assigned_values[
        score_order
    ] = sorted_burden_values


    return assigned_values



def generate_mst_instance(
    n_items: int,
    regime: str,
    seed: int,
    regime_strength: float = REGIME_STRENGTH,
) -> list[mst_core.InformationItem]:
    """
    Generate one reproducible synthetic MST benchmark
    instance with regime-invariant marginal distributions.

    The three regimes use identical marginal distributions
    for utility, privacy exposure, and interpretation cost.
    They differ only in the association between utility and
    disclosure burden.

    Parameters
    ----------
    n_items
        Number of privacy-admissible candidate items.

    regime
        One of:
        - independent
        - utility_burden_conflict
        - utility_burden_alignment

    seed
        Random seed for reproducibility.

    regime_strength
        Strength of the noisy rank association in [0, 1].
        Larger values create stronger positive or negative
        utility–burden relationships.

    Returns
    -------
    list[mst_core.InformationItem]
        Generated privacy-admissible candidate items.
    """

    if n_items <= 0:
        raise ValueError(
            "n_items must be greater than zero."
        )

    if regime not in INSTANCE_REGIMES:
        raise ValueError(
            f"Unknown regime: {regime}"
        )

    if not 0.0 <= regime_strength <= 1.0:
        raise ValueError(
            "regime_strength must be in [0, 1]."
        )


    rng = np.random.default_rng(
        seed
    )


    # ----------------------------------------------
    # Modeled item utility
    # ----------------------------------------------

    utility_latent = rng.beta(
        a=2.0,
        b=2.0,
        size=n_items,
    )

    utility_values = (
        0.08
        +
        0.52
        * utility_latent
    )


    # ----------------------------------------------
    # Regime-invariant burden marginals
    # ----------------------------------------------

    base_privacy_latent = rng.beta(
        a=2.0,
        b=2.0,
        size=n_items,
    )

    base_cognitive_latent = rng.beta(
        a=2.0,
        b=2.0,
        size=n_items,
    )


    # ----------------------------------------------
    # Change association without changing marginals
    # ----------------------------------------------

    privacy_latent = (
        _assign_burden_preserving_marginal(
            base_values=
                base_privacy_latent,

            utility_values=
                utility_latent,

            regime=
                regime,

            rng=
                rng,

            regime_strength=
                regime_strength,
        )
    )


    cognitive_latent = (
        _assign_burden_preserving_marginal(
            base_values=
                base_cognitive_latent,

            utility_values=
                utility_latent,

            regime=
                regime,

            rng=
                rng,

            regime_strength=
                regime_strength,
        )
    )


    # ----------------------------------------------
    # Convert to benchmark scales
    # ----------------------------------------------

    privacy_values = (
        0.01
        +
        0.24
        * privacy_latent
    )

    cognitive_values = (
        0.10
        +
        0.90
        * cognitive_latent
    )


    # ----------------------------------------------
    # Construct information items
    # ----------------------------------------------

    items = []

    for index in range(
        n_items
    ):

        items.append(
            mst_core.InformationItem(
                item_id=
                    f"z{index + 1:03d}",

                name=(
                    "Synthetic information item "
                    f"{index + 1}"
                ),

                utility=float(
                    utility_values[
                        index
                    ]
                ),

                privacy=float(
                    privacy_values[
                        index
                    ]
                ),

                cognitive_cost=float(
                    cognitive_values[
                        index
                    ]
                ),
            )
        )


    return items


print(
    "Marginal-preserving MST generator defined."
)

print(
    "Default regime strength:",
    REGIME_STRENGTH
)

# ===== notebook 02, code cell 82 =====
print('### notebook cell 82', flush=True)
import os
PROJECT_ROOT = os.environ.get("MST_PROJECT_ROOT", "..")
# --------------------------------------------------
# Step 124 — Coverage-based submodular utility:
# matched marginal-preserving generator validation
# --------------------------------------------------

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

COVERAGE_GENERATOR_VERSION = (
    "coverage_submodular_marginal_preserving_v1"
)

COVERAGE_VALIDATION_SEED = (
    globals().get(
        "GLOBAL_SEED",
        2026,
    )
    + 124
)

COVERAGE_VALIDATION_INSTANCES = 500
COVERAGE_VALIDATION_N_ITEMS = 64
COVERAGE_FEATURES = 12

# Probability that an item provides nonzero coverage
# for a particular decision-relevant feature.
COVERAGE_DENSITY = 0.35

# Controls the noisy rank association between
# singleton utility and burden.
COVERAGE_REGIME_STRENGTH = 0.75


if "RESULTS_DIR" not in globals():

    RESULTS_DIR = Path(
        os.path.join(PROJECT_ROOT, "05_Results")
    )


if "generate_mst_instance" not in globals():

    raise NameError(
        "generate_mst_instance is unavailable. "
        "Run the corrected marginal-preserving-v2 "
        "generator cell first."
    )


COVERAGE_VALIDATION_RAW_PATH = (
    RESULTS_DIR
    / "coverage_generator_validation_raw.csv"
)


COVERAGE_VALIDATION_SUMMARY_PATH = (
    RESULTS_DIR
    / "coverage_generator_validation_summary.csv"
)


COVERAGE_VALIDATION_METADATA_PATH = (
    RESULTS_DIR
    / "coverage_generator_metadata.csv"
)


COVERAGE_REGIME_ORDER = [
    "independent",
    "utility_burden_alignment",
    "utility_burden_conflict",
]


# --------------------------------------------------
# Coverage item representation
# --------------------------------------------------

@dataclass(frozen=True)
class CoverageInformationItem:
    """
    Information item for the coverage-based
    submodular benchmark.
    """

    item_id: str

    privacy: float

    cognitive_cost: float

    feature_coverage: tuple


# --------------------------------------------------
# Coverage-based utility
# --------------------------------------------------

def coverage_utility(
    selected_items,
    feature_weights,
):
    """
    Weighted maximum coverage:

        U(S) = sum_f w_f max_{z in S} a_zf / sum_f w_f

    This function is normalized, monotone, and
    submodular for nonnegative feature weights and
    coverage values.
    """

    feature_weights = np.asarray(
        feature_weights,
        dtype=float,
    )


    if feature_weights.ndim != 1:

        raise ValueError(
            "feature_weights must be one-dimensional."
        )


    if np.any(
        feature_weights < 0.0
    ):

        raise ValueError(
            "feature_weights must be nonnegative."
        )


    total_feature_weight = (
        feature_weights.sum()
    )


    if total_feature_weight <= 0.0:

        raise ValueError(
            "At least one feature weight must be positive."
        )


    if len(
        selected_items
    ) == 0:

        return 0.0


    coverage_matrix = np.asarray(
        [
            item.feature_coverage
            for item in selected_items
        ],
        dtype=float,
    )


    maximum_feature_coverage = (
        coverage_matrix.max(
            axis=0
        )
    )


    return float(
        np.dot(
            feature_weights,
            maximum_feature_coverage,
        )
        /
        total_feature_weight
    )


# --------------------------------------------------
# Coverage matrix generation
# --------------------------------------------------

def generate_coverage_structure(
    n_items,
    n_features,
    seed,
    coverage_density,
):
    """
    Generate matched feature weights and an
    item-by-feature coverage matrix.

    Each item covers at least one feature, and each
    feature is covered by at least one item.
    """

    rng = np.random.default_rng(
        seed
    )


    # Unequal but bounded decision-feature importance.
    feature_weights = rng.lognormal(
        mean=0.0,
        sigma=0.45,
        size=n_features,
    )


    feature_weights = (
        feature_weights
        /
        feature_weights.sum()
    )


    active_mask = (
        rng.random(
            (
                n_items,
                n_features,
            )
        )
        <
        coverage_density
    )


    # Ensure every item carries some decision-relevant
    # information.
    empty_item_indices = np.where(
        active_mask.sum(
            axis=1
        )
        == 0
    )[0]


    for item_index in empty_item_indices:

        selected_feature = rng.integers(
            low=0,
            high=n_features,
        )


        active_mask[
            item_index,
            selected_feature,
        ] = True


    # Ensure every feature appears in the candidate
    # information pool.
    empty_feature_indices = np.where(
        active_mask.sum(
            axis=0
        )
        == 0
    )[0]


    for feature_index in empty_feature_indices:

        selected_item = rng.integers(
            low=0,
            high=n_items,
        )


        active_mask[
            selected_item,
            feature_index,
        ] = True


    coverage_strengths = rng.beta(
        a=2.5,
        b=1.5,
        size=(
            n_items,
            n_features,
        ),
    )


    coverage_matrix = np.where(
        active_mask,
        coverage_strengths,
        0.0,
    )


    return (
        feature_weights,
        coverage_matrix,
    )


# --------------------------------------------------
# Marginal-preserving burden assignment
# --------------------------------------------------

def noisy_rank_assignment(
    burden_values,
    utility_proxy,
    association_direction,
    strength,
    rng,
):
    """
    Reassign an unchanged burden-value multiset using
    a noisy utility-rank score.

    association_direction:
         0 -> independent
        +1 -> utility-burden conflict
        -1 -> utility-burden alignment
    """

    burden_values = np.asarray(
        burden_values,
        dtype=float,
    )


    utility_proxy = np.asarray(
        utility_proxy,
        dtype=float,
    )


    if len(
        burden_values
    ) != len(
        utility_proxy
    ):

        raise ValueError(
            "Burden and utility arrays must have "
            "the same length."
        )


    n_items = len(
        burden_values
    )


    sorted_burden_values = np.sort(
        burden_values
    )


    if association_direction == 0:

        assignment_order = rng.permutation(
            n_items
        )


    else:

        utility_ranks = pd.Series(
            utility_proxy
        ).rank(
            method="average"
        ).to_numpy(
            dtype=float
        )


        utility_rank_z = (
            utility_ranks
            -
            utility_ranks.mean()
        )


        utility_rank_sd = (
            utility_rank_z.std(
                ddof=0
            )
        )


        if utility_rank_sd > 0.0:

            utility_rank_z = (
                utility_rank_z
                /
                utility_rank_sd
            )


        random_noise = rng.normal(
            loc=0.0,
            scale=1.0,
            size=n_items,
        )


        latent_score = (
            association_direction
            *
            strength
            *
            utility_rank_z
            +
            np.sqrt(
                max(
                    0.0,
                    1.0
                    -
                    strength ** 2,
                )
            )
            *
            random_noise
        )


        assignment_order = np.argsort(
            latent_score,
            kind="mergesort",
        )


    assigned_values = np.empty(
        n_items,
        dtype=float,
    )


    assigned_values[
        assignment_order
    ] = sorted_burden_values


    return assigned_values


# --------------------------------------------------
# Matched coverage instance generator
# --------------------------------------------------

def generate_matched_coverage_instances(
    n_items,
    n_features,
    seed,
    coverage_density=
        COVERAGE_DENSITY,
    regime_strength=
        COVERAGE_REGIME_STRENGTH,
):
    """
    Generate three matched structural regimes.

    Across regimes, the following are identical:
    - feature weights;
    - item-feature coverage values;
    - singleton utility values;
    - sorted privacy values;
    - sorted cognitive-cost values.

    Only the association between singleton utility and
    burden changes.
    """

    feature_weights, coverage_matrix = (
        generate_coverage_structure(
            n_items=n_items,
            n_features=n_features,
            seed=seed,
            coverage_density=
                coverage_density,
        )
    )


    singleton_utilities = (
        coverage_matrix
        @
        feature_weights
    )


    # Reuse the corrected-v2 burden distributions so
    # that the two utility benchmark families remain
    # comparable in their burden marginals.
    burden_source_items = (
        generate_mst_instance(
            n_items=n_items,
            regime="independent",
            seed=seed,
        )
    )


    base_privacy_values = np.asarray(
        [
            float(
                item.privacy
            )
            for item in burden_source_items
        ],
        dtype=float,
    )


    base_cognitive_values = np.asarray(
        [
            float(
                item.cognitive_cost
            )
            for item in burden_source_items
        ],
        dtype=float,
    )


    regime_directions = {
        "independent":
            0,

        "utility_burden_alignment":
            -1,

        "utility_burden_conflict":
            1,
    }


    matched_instances = {}


    for regime_index, regime in enumerate(
        COVERAGE_REGIME_ORDER
    ):

        direction = regime_directions[
            regime
        ]


        privacy_rng = np.random.default_rng(
            seed
            +
            10_000
            +
            100
            *
            regime_index
        )


        cognitive_rng = np.random.default_rng(
            seed
            +
            20_000
            +
            100
            *
            regime_index
        )


        assigned_privacy = (
            noisy_rank_assignment(
                burden_values=
                    base_privacy_values,

                utility_proxy=
                    singleton_utilities,

                association_direction=
                    direction,

                strength=
                    regime_strength,

                rng=
                    privacy_rng,
            )
        )


        assigned_cognitive = (
            noisy_rank_assignment(
                burden_values=
                    base_cognitive_values,

                utility_proxy=
                    singleton_utilities,

                association_direction=
                    direction,

                strength=
                    regime_strength,

                rng=
                    cognitive_rng,
            )
        )


        items = [
            CoverageInformationItem(
                item_id=
                    f"c{item_index + 1:03d}",

                privacy=
                    float(
                        assigned_privacy[
                            item_index
                        ]
                    ),

                cognitive_cost=
                    float(
                        assigned_cognitive[
                            item_index
                        ]
                    ),

                feature_coverage=
                    tuple(
                        float(value)
                        for value in coverage_matrix[
                            item_index
                        ]
                    ),
            )
            for item_index in range(
                n_items
            )
        ]


        matched_instances[
            regime
        ] = {
            "items":
                items,

            "feature_weights":
                feature_weights.copy(),

            "coverage_matrix":
                coverage_matrix.copy(),

            "singleton_utilities":
                singleton_utilities.copy(),
        }


    return matched_instances


# --------------------------------------------------
# Validation run
# --------------------------------------------------

validation_rows = []
marginal_audit_rows = []


for replicate in range(
    COVERAGE_VALIDATION_INSTANCES
):

    matched_seed = (
        COVERAGE_VALIDATION_SEED
        +
        1_000_000
        +
        replicate
    )


    matched_instances = (
        generate_matched_coverage_instances(
            n_items=
                COVERAGE_VALIDATION_N_ITEMS,

            n_features=
                COVERAGE_FEATURES,

            seed=
                matched_seed,
        )
    )


    reference_regime_data = (
        matched_instances[
            "independent"
        ]
    )


    reference_coverage_matrix = (
        reference_regime_data[
            "coverage_matrix"
        ]
    )


    reference_feature_weights = (
        reference_regime_data[
            "feature_weights"
        ]
    )


    reference_singleton_utility = (
        reference_regime_data[
            "singleton_utilities"
        ]
    )


    full_set_utility = coverage_utility(
        selected_items=
            reference_regime_data[
                "items"
            ],

        feature_weights=
            reference_feature_weights,
    )


    nonzero_coverage_fraction = (
        reference_coverage_matrix
        >
        0.0
    ).mean()


    privacy_values_by_regime = {}
    cognitive_values_by_regime = {}


    for regime in COVERAGE_REGIME_ORDER:

        regime_data = matched_instances[
            regime
        ]


        regime_items = regime_data[
            "items"
        ]


        privacy_values = np.asarray(
            [
                item.privacy
                for item in regime_items
            ],
            dtype=float,
        )


        cognitive_values = np.asarray(
            [
                item.cognitive_cost
                for item in regime_items
            ],
            dtype=float,
        )


        privacy_values_by_regime[
            regime
        ] = privacy_values


        cognitive_values_by_regime[
            regime
        ] = cognitive_values


        utility_series = pd.Series(
            reference_singleton_utility
        )


        privacy_series = pd.Series(
            privacy_values
        )


        cognitive_series = pd.Series(
            cognitive_values
        )


        validation_rows.append(
            {
                "generator_version":
                    COVERAGE_GENERATOR_VERSION,

                "replicate":
                    replicate,

                "matched_seed":
                    matched_seed,

                "n_items":
                    COVERAGE_VALIDATION_N_ITEMS,

                "n_features":
                    COVERAGE_FEATURES,

                "regime":
                    regime,

                "regime_strength":
                    COVERAGE_REGIME_STRENGTH,

                "coverage_density_parameter":
                    COVERAGE_DENSITY,

                "observed_nonzero_coverage_fraction":
                    nonzero_coverage_fraction,

                "full_set_utility":
                    full_set_utility,

                "mean_singleton_utility":
                    reference_singleton_utility.mean(),

                "sd_singleton_utility":
                    reference_singleton_utility.std(
                        ddof=1
                    ),

                "utility_privacy_pearson":
                    utility_series.corr(
                        privacy_series,
                        method="pearson",
                    ),

                "utility_privacy_spearman":
                    utility_series.corr(
                        privacy_series,
                        method="spearman",
                    ),

                "utility_cognitive_pearson":
                    utility_series.corr(
                        cognitive_series,
                        method="pearson",
                    ),

                "utility_cognitive_spearman":
                    utility_series.corr(
                        cognitive_series,
                        method="spearman",
                    ),

                "mean_privacy":
                    privacy_values.mean(),

                "sd_privacy":
                    privacy_values.std(
                        ddof=1
                    ),

                "mean_cognitive":
                    cognitive_values.mean(),

                "sd_cognitive":
                    cognitive_values.std(
                        ddof=1
                    ),
            }
        )


    independent_privacy = np.sort(
        privacy_values_by_regime[
            "independent"
        ]
    )


    independent_cognitive = np.sort(
        cognitive_values_by_regime[
            "independent"
        ]
    )


    maximum_privacy_marginal_difference = 0.0
    maximum_cognitive_marginal_difference = 0.0


    for regime in COVERAGE_REGIME_ORDER:

        maximum_privacy_marginal_difference = max(
            maximum_privacy_marginal_difference,

            float(
                np.max(
                    np.abs(
                        np.sort(
                            privacy_values_by_regime[
                                regime
                            ]
                        )
                        -
                        independent_privacy
                    )
                )
            ),
        )


        maximum_cognitive_marginal_difference = max(
            maximum_cognitive_marginal_difference,

            float(
                np.max(
                    np.abs(
                        np.sort(
                            cognitive_values_by_regime[
                                regime
                            ]
                        )
                        -
                        independent_cognitive
                    )
                )
            ),
        )


    marginal_audit_rows.append(
        {
            "replicate":
                replicate,

            "matched_seed":
                matched_seed,

            "maximum_sorted_privacy_difference":
                maximum_privacy_marginal_difference,

            "maximum_sorted_cognitive_difference":
                maximum_cognitive_marginal_difference,
        }
    )


# --------------------------------------------------
# Summaries
# --------------------------------------------------

coverage_validation_raw_df = pd.DataFrame(
    validation_rows
)


coverage_marginal_audit_df = pd.DataFrame(
    marginal_audit_rows
)


coverage_validation_summary_df = (
    coverage_validation_raw_df
    .groupby(
        "regime",
        as_index=False,
    )
    .agg(
        instances=(
            "replicate",
            "nunique",
        ),

        mean_utility_privacy_pearson=(
            "utility_privacy_pearson",
            "mean",
        ),

        mean_utility_privacy_spearman=(
            "utility_privacy_spearman",
            "mean",
        ),

        mean_utility_cognitive_pearson=(
            "utility_cognitive_pearson",
            "mean",
        ),

        mean_utility_cognitive_spearman=(
            "utility_cognitive_spearman",
            "mean",
        ),

        mean_singleton_utility=(
            "mean_singleton_utility",
            "mean",
        ),

        mean_full_set_utility=(
            "full_set_utility",
            "mean",
        ),

        minimum_full_set_utility=(
            "full_set_utility",
            "min",
        ),

        mean_privacy=(
            "mean_privacy",
            "mean",
        ),

        mean_privacy_sd=(
            "sd_privacy",
            "mean",
        ),

        mean_cognitive=(
            "mean_cognitive",
            "mean",
        ),

        mean_cognitive_sd=(
            "sd_cognitive",
            "mean",
        ),

        mean_observed_coverage_density=(
            "observed_nonzero_coverage_fraction",
            "mean",
        ),
    )
)


coverage_metadata_df = pd.DataFrame(
    [
        {
            "generator_version":
                COVERAGE_GENERATOR_VERSION,

            "validation_seed":
                COVERAGE_VALIDATION_SEED,

            "validation_instances":
                COVERAGE_VALIDATION_INSTANCES,

            "validation_n_items":
                COVERAGE_VALIDATION_N_ITEMS,

            "n_features":
                COVERAGE_FEATURES,

            "coverage_density":
                COVERAGE_DENSITY,

            "regime_strength":
                COVERAGE_REGIME_STRENGTH,

            "utility_function":
                (
                    "weighted maximum feature coverage"
                ),

            "burden_distribution_source":
                (
                    "corrected marginal-preserving-v2 "
                    "independent generator"
                ),

            "privacy_marginals_preserved":
                True,

            "cognitive_marginals_preserved":
                True,

            "regime_association_proxy":
                "singleton coverage utility",
        }
    ]
)


# --------------------------------------------------
# Integrity assertions
# --------------------------------------------------

assert len(
    coverage_validation_raw_df
) == (
    COVERAGE_VALIDATION_INSTANCES
    *
    len(
        COVERAGE_REGIME_ORDER
    )
)


assert (
    coverage_validation_summary_df[
        "instances"
    ]
    ==
    COVERAGE_VALIDATION_INSTANCES
).all()


assert (
    coverage_marginal_audit_df[
        "maximum_sorted_privacy_difference"
    ].max()
    <= 1e-15
)


assert (
    coverage_marginal_audit_df[
        "maximum_sorted_cognitive_difference"
    ].max()
    <= 1e-15
)


assert (
    coverage_validation_raw_df[
        "full_set_utility"
    ]
    >= 0.0
).all()


assert (
    coverage_validation_raw_df[
        "full_set_utility"
    ]
    <= 1.0
    +
    1e-12
).all()


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

coverage_validation_raw_df.to_csv(
    COVERAGE_VALIDATION_RAW_PATH,
    index=False,
)


coverage_validation_summary_df.to_csv(
    COVERAGE_VALIDATION_SUMMARY_PATH,
    index=False,
)


coverage_metadata_df.to_csv(
    COVERAGE_VALIDATION_METADATA_PATH,
    index=False,
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print(
    "======================================"
)

print(
    "COVERAGE-BASED GENERATOR VALIDATION"
)

print(
    "======================================"
)


print(
    coverage_validation_summary_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.6f}",
    )
)


print(
    "\n======================================"
)

print(
    "MATCHED-MARGINAL AUDIT"
)

print(
    "======================================"
)


print(
    "Maximum sorted privacy difference:"
)

print(
    f"{coverage_marginal_audit_df['maximum_sorted_privacy_difference'].max():.16f}"
)


print(
    "\nMaximum sorted cognitive difference:"
)

print(
    f"{coverage_marginal_audit_df['maximum_sorted_cognitive_difference'].max():.16f}"
)


print(
    "\nCoverage validation raw data saved to:"
)

print(
    COVERAGE_VALIDATION_RAW_PATH
)


print(
    "\nCoverage validation summary saved to:"
)

print(
    COVERAGE_VALIDATION_SUMMARY_PATH
)


print(
    "\nCoverage generator metadata saved to:"
)

print(
    COVERAGE_VALIDATION_METADATA_PATH
)

# ===== notebook 02, code cell 84 =====
print('### notebook cell 84', flush=True)
import os
PROJECT_ROOT = os.environ.get("MST_PROJECT_ROOT", "..")
# --------------------------------------------------
# Step 126 — Candidate-pool-normalized coverage
# utility calibration and submodularity audit
# --------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

COVERAGE_POOL_NORMALIZED_VERSION = (
    "coverage_submodular_pool_normalized_v2"
)


COVERAGE_POOL_NORMALIZED_BASE_SEED = (
    globals().get(
        "COVERAGE_FEASIBILITY_BASE_SEED",
        2026 + 125,
    )
)


COVERAGE_POOL_NORMALIZED_SIZES = [
    8,
    10,
    12,
    14,
    16,
]


COVERAGE_POOL_NORMALIZED_THRESHOLDS = [
    0.70,
    0.80,
    0.90,
    0.95,
]


COVERAGE_POOL_NORMALIZED_REPLICATES = 100

COVERAGE_PROPERTY_TESTS_PER_GROUP = 25

COVERAGE_PROPERTY_TOLERANCE = 1e-12


COVERAGE_POOL_NORMALIZED_FEATURES = (
    globals().get(
        "COVERAGE_FEATURES",
        12,
    )
)


COVERAGE_POOL_NORMALIZED_DENSITY = (
    globals().get(
        "COVERAGE_DENSITY",
        0.35,
    )
)


COVERAGE_POOL_NORMALIZED_REGIME_STRENGTH = (
    globals().get(
        "COVERAGE_REGIME_STRENGTH",
        0.75,
    )
)


if "RESULTS_DIR" not in globals():

    RESULTS_DIR = Path(
        os.path.join(PROJECT_ROOT, "05_Results")
    )


if "TABLES_DIR" not in globals():

    TABLES_DIR = Path(
        os.path.join(PROJECT_ROOT, "07_Tables")
    )


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if "generate_matched_coverage_instances" not in globals():

    raise NameError(
        "generate_matched_coverage_instances is unavailable. "
        "Run Step 124 first."
    )


# --------------------------------------------------
# Output paths
# --------------------------------------------------

POOL_NORMALIZED_RAW_PATH = (
    RESULTS_DIR
    / "coverage_pool_normalized_calibration_raw.csv"
)


POOL_NORMALIZED_UTILITY_SUMMARY_PATH = (
    TABLES_DIR
    / "coverage_pool_normalized_utility_by_candidate_size.csv"
)


POOL_NORMALIZED_FEASIBILITY_PATH = (
    TABLES_DIR
    / "coverage_pool_normalized_feasibility_by_size_threshold.csv"
)


POOL_NORMALIZED_PROPERTY_AUDIT_PATH = (
    RESULTS_DIR
    / "coverage_pool_normalized_submodularity_audit.csv"
)


POOL_NORMALIZED_PROPERTY_SUMMARY_PATH = (
    TABLES_DIR
    / "coverage_pool_normalized_property_audit_summary.csv"
)


POOL_NORMALIZED_METADATA_PATH = (
    RESULTS_DIR
    / "coverage_pool_normalized_metadata.csv"
)


# --------------------------------------------------
# Candidate-pool-normalized coverage utility
# --------------------------------------------------

def pool_normalized_coverage_utility(
    selected_items,
    all_items,
    feature_weights,
):
    """
    Candidate-pool-normalized weighted maximum coverage:

        U_D(S) =
            sum_f w_f max_{z in S} a_zf
            --------------------------------
            sum_f w_f max_{z in D} a_zf

    where D is the complete candidate information pool.

    Properties:
    - U_D(empty) = 0
    - U_D(D) = 1
    - monotone
    - submodular
    """

    feature_weights = np.asarray(
        feature_weights,
        dtype=float,
    )


    if feature_weights.ndim != 1:

        raise ValueError(
            "feature_weights must be one-dimensional."
        )


    if np.any(
        feature_weights < 0.0
    ):

        raise ValueError(
            "feature_weights must be nonnegative."
        )


    if len(
        all_items
    ) == 0:

        raise ValueError(
            "all_items must contain at least one item."
        )


    all_coverage_matrix = np.asarray(
        [
            item.feature_coverage
            for item in all_items
        ],
        dtype=float,
    )


    if (
        all_coverage_matrix.ndim
        != 2
    ):

        raise ValueError(
            "The candidate-pool coverage matrix must "
            "be two-dimensional."
        )


    if (
        all_coverage_matrix.shape[1]
        != len(
            feature_weights
        )
    ):

        raise ValueError(
            "Coverage dimensions do not match "
            "feature_weights."
        )


    available_feature_coverage = (
        all_coverage_matrix.max(
            axis=0
        )
    )


    available_weighted_coverage = float(
        np.dot(
            feature_weights,
            available_feature_coverage,
        )
    )


    if available_weighted_coverage <= 0.0:

        raise ValueError(
            "The candidate pool provides no positive "
            "weighted feature coverage."
        )


    if len(
        selected_items
    ) == 0:

        return 0.0


    selected_coverage_matrix = np.asarray(
        [
            item.feature_coverage
            for item in selected_items
        ],
        dtype=float,
    )


    selected_feature_coverage = (
        selected_coverage_matrix.max(
            axis=0
        )
    )


    selected_weighted_coverage = float(
        np.dot(
            feature_weights,
            selected_feature_coverage,
        )
    )


    utility = (
        selected_weighted_coverage
        /
        available_weighted_coverage
    )


    # Remove negligible floating-point excursions.
    if (
        utility < 0.0
        and
        utility >= -1e-12
    ):

        utility = 0.0


    if (
        utility > 1.0
        and
        utility <= 1.0 + 1e-12
    ):

        utility = 1.0


    return float(
        utility
    )


# --------------------------------------------------
# Vectorized singleton utility calculation
# --------------------------------------------------

def pool_normalized_singleton_utilities(
    all_items,
    feature_weights,
):
    """
    Return U_D({z}) for every candidate item.
    """

    feature_weights = np.asarray(
        feature_weights,
        dtype=float,
    )


    coverage_matrix = np.asarray(
        [
            item.feature_coverage
            for item in all_items
        ],
        dtype=float,
    )


    available_feature_coverage = (
        coverage_matrix.max(
            axis=0
        )
    )


    denominator = float(
        np.dot(
            feature_weights,
            available_feature_coverage,
        )
    )


    if denominator <= 0.0:

        raise ValueError(
            "The candidate pool has zero attainable "
            "weighted coverage."
        )


    singleton_numerators = (
        coverage_matrix
        @
        feature_weights
    )


    return (
        singleton_numerators
        /
        denominator
    )


# --------------------------------------------------
# Raw ideal-denominator utility for comparison
# --------------------------------------------------

def ideal_normalized_coverage_utility(
    selected_items,
    feature_weights,
):
    """
    Original coverage utility normalized against
    ideal unit coverage of every feature.
    """

    feature_weights = np.asarray(
        feature_weights,
        dtype=float,
    )


    if len(
        selected_items
    ) == 0:

        return 0.0


    selected_matrix = np.asarray(
        [
            item.feature_coverage
            for item in selected_items
        ],
        dtype=float,
    )


    selected_maximum = (
        selected_matrix.max(
            axis=0
        )
    )


    return float(
        np.dot(
            feature_weights,
            selected_maximum,
        )
        /
        feature_weights.sum()
    )


# --------------------------------------------------
# Calibration and property-audit containers
# --------------------------------------------------

calibration_rows = []
feasibility_rows = []
property_rows = []


# --------------------------------------------------
# Generate matched benchmark groups
# --------------------------------------------------

for n_items in COVERAGE_POOL_NORMALIZED_SIZES:

    for replicate in range(
        COVERAGE_POOL_NORMALIZED_REPLICATES
    ):

        matched_seed = (
            72_000_000
            +
            100_000
            *
            n_items
            +
            COVERAGE_POOL_NORMALIZED_BASE_SEED
            +
            replicate
        )


        matched_instances = (
            generate_matched_coverage_instances(
                n_items=n_items,

                n_features=
                    COVERAGE_POOL_NORMALIZED_FEATURES,

                seed=
                    matched_seed,

                coverage_density=
                    COVERAGE_POOL_NORMALIZED_DENSITY,

                regime_strength=
                    COVERAGE_POOL_NORMALIZED_REGIME_STRENGTH,
            )
        )


        # Utility structure is identical across the
        # three matched burden-association regimes.
        reference_data = (
            matched_instances[
                "independent"
            ]
        )


        all_items = reference_data[
            "items"
        ]


        feature_weights = np.asarray(
            reference_data[
                "feature_weights"
            ],
            dtype=float,
        )


        raw_full_set_utility = (
            ideal_normalized_coverage_utility(
                selected_items=
                    all_items,

                feature_weights=
                    feature_weights,
            )
        )


        normalized_full_set_utility = (
            pool_normalized_coverage_utility(
                selected_items=
                    all_items,

                all_items=
                    all_items,

                feature_weights=
                    feature_weights,
            )
        )


        singleton_utilities = (
            pool_normalized_singleton_utilities(
                all_items=
                    all_items,

                feature_weights=
                    feature_weights,
            )
        )


        calibration_rows.append(
            {
                "generator_version":
                    COVERAGE_POOL_NORMALIZED_VERSION,

                "matched_group_id":
                    (
                        f"COVN_M{n_items}_"
                        f"R{replicate:03d}"
                    ),

                "n_items":
                    n_items,

                "replicate":
                    replicate,

                "matched_seed":
                    matched_seed,

                "n_features":
                    COVERAGE_POOL_NORMALIZED_FEATURES,

                "coverage_density":
                    COVERAGE_POOL_NORMALIZED_DENSITY,

                "raw_full_set_utility":
                    raw_full_set_utility,

                "pool_normalized_full_set_utility":
                    normalized_full_set_utility,

                "mean_pool_normalized_singleton_utility":
                    float(
                        singleton_utilities.mean()
                    ),

                "minimum_pool_normalized_singleton_utility":
                    float(
                        singleton_utilities.min()
                    ),

                "maximum_pool_normalized_singleton_utility":
                    float(
                        singleton_utilities.max()
                    ),
            }
        )


        for tau in (
            COVERAGE_POOL_NORMALIZED_THRESHOLDS
        ):

            feasibility_rows.append(
                {
                    "matched_group_id":
                        (
                            f"COVN_M{n_items}_"
                            f"R{replicate:03d}"
                        ),

                    "n_items":
                        n_items,

                    "replicate":
                        replicate,

                    "matched_seed":
                        matched_seed,

                    "tau":
                        tau,

                    "pool_normalized_full_set_utility":
                        normalized_full_set_utility,

                    "is_feasible":
                        int(
                            normalized_full_set_utility
                            +
                            COVERAGE_PROPERTY_TOLERANCE
                            >= tau
                        ),
                }
            )


        # ------------------------------------------
        # Random monotonicity and diminishing-returns
        # checks
        # ------------------------------------------

        property_rng = np.random.default_rng(
            matched_seed
            +
            50_000_000
        )


        for property_test in range(
            COVERAGE_PROPERTY_TESTS_PER_GROUP
        ):

            permutation = (
                property_rng.permutation(
                    n_items
                )
            )


            # A has at most n-2 elements so B can
            # retain at least one candidate x outside.
            a_size = int(
                property_rng.integers(
                    low=0,
                    high=n_items - 1,
                )
            )


            b_size = int(
                property_rng.integers(
                    low=a_size,
                    high=n_items,
                )
            )


            a_indices = permutation[
                :a_size
            ]


            b_indices = permutation[
                :b_size
            ]


            x_index = int(
                permutation[
                    b_size
                ]
            )


            a_items = [
                all_items[
                    item_index
                ]
                for item_index in a_indices
            ]


            b_items = [
                all_items[
                    item_index
                ]
                for item_index in b_indices
            ]


            x_item = all_items[
                x_index
            ]


            utility_a = (
                pool_normalized_coverage_utility(
                    selected_items=
                        a_items,

                    all_items=
                        all_items,

                    feature_weights=
                        feature_weights,
                )
            )


            utility_b = (
                pool_normalized_coverage_utility(
                    selected_items=
                        b_items,

                    all_items=
                        all_items,

                    feature_weights=
                        feature_weights,
                )
            )


            utility_a_plus_x = (
                pool_normalized_coverage_utility(
                    selected_items=
                        a_items
                        +
                        [
                            x_item
                        ],

                    all_items=
                        all_items,

                    feature_weights=
                        feature_weights,
                )
            )


            utility_b_plus_x = (
                pool_normalized_coverage_utility(
                    selected_items=
                        b_items
                        +
                        [
                            x_item
                        ],

                    all_items=
                        all_items,

                    feature_weights=
                        feature_weights,
                )
            )


            marginal_gain_a = (
                utility_a_plus_x
                -
                utility_a
            )


            marginal_gain_b = (
                utility_b_plus_x
                -
                utility_b
            )


            monotonicity_violation = max(
                0.0,
                utility_a
                -
                utility_b,
            )


            diminishing_returns_violation = max(
                0.0,
                marginal_gain_b
                -
                marginal_gain_a,
            )


            range_violation = max(
                0.0,
                -utility_a,
                utility_a
                -
                1.0,
                -utility_b,
                utility_b
                -
                1.0,
                -utility_a_plus_x,
                utility_a_plus_x
                -
                1.0,
                -utility_b_plus_x,
                utility_b_plus_x
                -
                1.0,
            )


            property_rows.append(
                {
                    "matched_group_id":
                        (
                            f"COVN_M{n_items}_"
                            f"R{replicate:03d}"
                        ),

                    "n_items":
                        n_items,

                    "replicate":
                        replicate,

                    "matched_seed":
                        matched_seed,

                    "property_test":
                        property_test,

                    "a_size":
                        a_size,

                    "b_size":
                        b_size,

                    "utility_a":
                        utility_a,

                    "utility_b":
                        utility_b,

                    "marginal_gain_a":
                        marginal_gain_a,

                    "marginal_gain_b":
                        marginal_gain_b,

                    "monotonicity_violation":
                        monotonicity_violation,

                    "diminishing_returns_violation":
                        diminishing_returns_violation,

                    "range_violation":
                        range_violation,
                }
            )


    print(
        "Completed pool-normalized coverage size:",
        n_items
    )


# --------------------------------------------------
# Construct data frames
# --------------------------------------------------

calibration_df = pd.DataFrame(
    calibration_rows
)


feasibility_df = pd.DataFrame(
    feasibility_rows
)


property_audit_df = pd.DataFrame(
    property_rows
)


# --------------------------------------------------
# Candidate-size utility summary
# --------------------------------------------------

utility_summary_df = (
    calibration_df
    .groupby(
        "n_items",
        as_index=False,
    )
    .agg(
        matched_groups=(
            "matched_group_id",
            "nunique",
        ),

        mean_raw_full_set_utility=(
            "raw_full_set_utility",
            "mean",
        ),

        minimum_raw_full_set_utility=(
            "raw_full_set_utility",
            "min",
        ),

        mean_pool_normalized_full_set_utility=(
            "pool_normalized_full_set_utility",
            "mean",
        ),

        minimum_pool_normalized_full_set_utility=(
            "pool_normalized_full_set_utility",
            "min",
        ),

        maximum_pool_normalized_full_set_utility=(
            "pool_normalized_full_set_utility",
            "max",
        ),

        mean_singleton_utility=(
            "mean_pool_normalized_singleton_utility",
            "mean",
        ),

        minimum_singleton_utility=(
            "minimum_pool_normalized_singleton_utility",
            "min",
        ),

        maximum_singleton_utility=(
            "maximum_pool_normalized_singleton_utility",
            "max",
        ),
    )
)


# --------------------------------------------------
# Feasibility summary
# --------------------------------------------------

feasibility_summary_df = (
    feasibility_df
    .groupby(
        [
            "n_items",
            "tau",
        ],
        as_index=False,
    )
    .agg(
        matched_groups=(
            "matched_group_id",
            "nunique",
        ),

        feasible_groups=(
            "is_feasible",
            "sum",
        ),

        minimum_full_set_utility=(
            "pool_normalized_full_set_utility",
            "min",
        ),

        maximum_full_set_utility=(
            "pool_normalized_full_set_utility",
            "max",
        ),
    )
)


feasibility_summary_df[
    "feasibility_percent"
] = (
    100.0
    *
    feasibility_summary_df[
        "feasible_groups"
    ]
    /
    feasibility_summary_df[
        "matched_groups"
    ]
)


# --------------------------------------------------
# Property-audit summary
# --------------------------------------------------

property_summary_df = pd.DataFrame(
    [
        {
            "matched_groups":
                property_audit_df[
                    "matched_group_id"
                ].nunique(),

            "random_property_tests":
                len(
                    property_audit_df
                ),

            "maximum_monotonicity_violation":
                property_audit_df[
                    "monotonicity_violation"
                ].max(),

            "maximum_diminishing_returns_violation":
                property_audit_df[
                    "diminishing_returns_violation"
                ].max(),

            "maximum_range_violation":
                property_audit_df[
                    "range_violation"
                ].max(),

            "maximum_full_set_deviation_from_one":
                (
                    calibration_df[
                        "pool_normalized_full_set_utility"
                    ]
                    -
                    1.0
                )
                .abs()
                .max(),

            "monotonicity_failures_above_tolerance":
                int(
                    (
                        property_audit_df[
                            "monotonicity_violation"
                        ]
                        >
                        COVERAGE_PROPERTY_TOLERANCE
                    ).sum()
                ),

            "diminishing_returns_failures_above_tolerance":
                int(
                    (
                        property_audit_df[
                            "diminishing_returns_violation"
                        ]
                        >
                        COVERAGE_PROPERTY_TOLERANCE
                    ).sum()
                ),

            "range_failures_above_tolerance":
                int(
                    (
                        property_audit_df[
                            "range_violation"
                        ]
                        >
                        COVERAGE_PROPERTY_TOLERANCE
                    ).sum()
                ),
        }
    ]
)


# --------------------------------------------------
# Integrity assertions
# --------------------------------------------------

expected_groups = (
    len(
        COVERAGE_POOL_NORMALIZED_SIZES
    )
    *
    COVERAGE_POOL_NORMALIZED_REPLICATES
)


expected_property_tests = (
    expected_groups
    *
    COVERAGE_PROPERTY_TESTS_PER_GROUP
)


assert len(
    calibration_df
) == expected_groups


assert len(
    property_audit_df
) == expected_property_tests


assert (
    calibration_df[
        "matched_group_id"
    ].nunique()
    == expected_groups
)


assert (
    calibration_df[
        "pool_normalized_full_set_utility"
    ]
    .sub(1.0)
    .abs()
    .max()
    <=
    COVERAGE_PROPERTY_TOLERANCE
)


assert (
    feasibility_summary_df[
        "feasibility_percent"
    ]
    ==
    100.0
).all()


assert (
    property_audit_df[
        "monotonicity_violation"
    ].max()
    <=
    COVERAGE_PROPERTY_TOLERANCE
)


assert (
    property_audit_df[
        "diminishing_returns_violation"
    ].max()
    <=
    COVERAGE_PROPERTY_TOLERANCE
)


assert (
    property_audit_df[
        "range_violation"
    ].max()
    <=
    COVERAGE_PROPERTY_TOLERANCE
)


# --------------------------------------------------
# Metadata
# --------------------------------------------------

metadata_df = pd.DataFrame(
    [
        {
            "utility_version":
                COVERAGE_POOL_NORMALIZED_VERSION,

            "candidate_sizes":
                repr(
                    COVERAGE_POOL_NORMALIZED_SIZES
                ),

            "replicates_per_size":
                COVERAGE_POOL_NORMALIZED_REPLICATES,

            "total_matched_groups":
                expected_groups,

            "thresholds":
                repr(
                    COVERAGE_POOL_NORMALIZED_THRESHOLDS
                ),

            "n_features":
                COVERAGE_POOL_NORMALIZED_FEATURES,

            "coverage_density":
                COVERAGE_POOL_NORMALIZED_DENSITY,

            "regime_strength":
                COVERAGE_POOL_NORMALIZED_REGIME_STRENGTH,

            "normalization_denominator":
                (
                    "weighted maximum feature coverage "
                    "available in the candidate pool"
                ),

            "threshold_interpretation":
                (
                    "fraction of available modeled "
                    "decision-support utility"
                ),

            "full_set_utility":
                1.0,

            "submodularity_preserved":
                True,

            "monotonicity_preserved":
                True,

            "property_tests_per_group":
                COVERAGE_PROPERTY_TESTS_PER_GROUP,

            "property_tolerance":
                COVERAGE_PROPERTY_TOLERANCE,
        }
    ]
)


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

calibration_df.to_csv(
    POOL_NORMALIZED_RAW_PATH,
    index=False,
)


utility_summary_df.to_csv(
    POOL_NORMALIZED_UTILITY_SUMMARY_PATH,
    index=False,
)


feasibility_summary_df.to_csv(
    POOL_NORMALIZED_FEASIBILITY_PATH,
    index=False,
)


property_audit_df.to_csv(
    POOL_NORMALIZED_PROPERTY_AUDIT_PATH,
    index=False,
)


property_summary_df.to_csv(
    POOL_NORMALIZED_PROPERTY_SUMMARY_PATH,
    index=False,
)


metadata_df.to_csv(
    POOL_NORMALIZED_METADATA_PATH,
    index=False,
)


# --------------------------------------------------
# Printed report
# --------------------------------------------------

print(
    "\n======================================"
)

print(
    "POOL-NORMALIZED COVERAGE UTILITY BY CANDIDATE SIZE"
)

print(
    "======================================"
)


print(
    utility_summary_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.6f}",
    )
)


print(
    "\n======================================"
)

print(
    "POOL-NORMALIZED COVERAGE FEASIBILITY"
)

print(
    "======================================"
)


print(
    feasibility_summary_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.6f}",
    )
)


print(
    "\n======================================"
)

print(
    "POOL-NORMALIZED COVERAGE PROPERTY AUDIT"
)

print(
    "======================================"
)


print(
    property_summary_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.16f}",
    )
)


print(
    "\nCalibration rows saved to:"
)

print(
    POOL_NORMALIZED_RAW_PATH
)


print(
    "\nUtility summary saved to:"
)

print(
    POOL_NORMALIZED_UTILITY_SUMMARY_PATH
)


print(
    "\nFeasibility summary saved to:"
)

print(
    POOL_NORMALIZED_FEASIBILITY_PATH
)


print(
    "\nProperty audit saved to:"
)

print(
    POOL_NORMALIZED_PROPERTY_AUDIT_PATH
)


print(
    "\nMetadata saved to:"
)

print(
    POOL_NORMALIZED_METADATA_PATH
)

# ===== notebook 02, code cell 85 =====
print('### notebook cell 85', flush=True)
import os
PROJECT_ROOT = os.environ.get("MST_PROJECT_ROOT", "..")
# --------------------------------------------------
# Step 127 — Pool-normalized coverage Exact-versus-
# Greedy matched-group smoke test
# --------------------------------------------------

from pathlib import Path
import itertools
import time

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

COVERAGE_SMOKE_N_ITEMS = 12

COVERAGE_SMOKE_SEED = (
    globals().get(
        "GLOBAL_SEED",
        2026,
    )
    + 127
)


COVERAGE_SMOKE_REGIMES = [
    "independent",
    "utility_burden_alignment",
    "utility_burden_conflict",
]


COVERAGE_SMOKE_THRESHOLDS = [
    0.70,
    0.80,
    0.90,
    0.95,
]


COVERAGE_SMOKE_WEIGHT_PROFILES = {
    "balanced": {
        "lambda_privacy": 1 / 3,
        "lambda_cognitive": 1 / 3,
        "lambda_cardinality": 1 / 3,
    },

    "privacy_priority": {
        "lambda_privacy": 0.60,
        "lambda_cognitive": 0.20,
        "lambda_cardinality": 0.20,
    },

    "interpretation_priority": {
        "lambda_privacy": 0.20,
        "lambda_cognitive": 0.60,
        "lambda_cardinality": 0.20,
    },

    "parsimony_priority": {
        "lambda_privacy": 0.20,
        "lambda_cognitive": 0.20,
        "lambda_cardinality": 0.60,
    },
}


COVERAGE_SMOKE_FEATURES = globals().get(
    "COVERAGE_FEATURES",
    12,
)


COVERAGE_SMOKE_DENSITY = globals().get(
    "COVERAGE_DENSITY",
    0.35,
)


COVERAGE_SMOKE_REGIME_STRENGTH = globals().get(
    "COVERAGE_REGIME_STRENGTH",
    0.75,
)


COVERAGE_NUMERICAL_TOLERANCE = 1e-12


if "RESULTS_DIR" not in globals():

    RESULTS_DIR = Path(
        os.path.join(PROJECT_ROOT, "05_Results")
    )


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if "generate_matched_coverage_instances" not in globals():

    raise NameError(
        "generate_matched_coverage_instances is unavailable. "
        "Run Step 124 first."
    )


if "pool_normalized_coverage_utility" not in globals():

    raise NameError(
        "pool_normalized_coverage_utility is unavailable. "
        "Run Step 126 first."
    )


COVERAGE_SMOKE_RESULTS_PATH = (
    RESULTS_DIR
    / "coverage_exact_vs_greedy_smoke_test.csv"
)


COVERAGE_SMOKE_AUDIT_PATH = (
    RESULTS_DIR
    / "coverage_exact_vs_greedy_smoke_test_audit.csv"
)


# --------------------------------------------------
# Subset evaluation
# --------------------------------------------------

def evaluate_coverage_subset(
    all_items,
    selected_indices,
    feature_weights,
    weights,
):
    """
    Evaluate one coverage-based information subset.
    """

    selected_indices = tuple(
        sorted(
            int(index)
            for index in selected_indices
        )
    )


    selected_items = [
        all_items[index]
        for index in selected_indices
    ]


    utility = pool_normalized_coverage_utility(
        selected_items=selected_items,
        all_items=all_items,
        feature_weights=feature_weights,
    )


    if selected_items:

        privacy = (
            1.0
            -
            np.prod(
                [
                    1.0
                    -
                    float(item.privacy)
                    for item in selected_items
                ]
            )
        )


        cognitive_cost = sum(
            float(item.cognitive_cost)
            for item in selected_items
        )


    else:

        privacy = 0.0
        cognitive_cost = 0.0


    full_cognitive_cost = sum(
        float(item.cognitive_cost)
        for item in all_items
    )


    if full_cognitive_cost > 0.0:

        normalized_cognitive_cost = (
            cognitive_cost
            /
            full_cognitive_cost
        )

    else:

        normalized_cognitive_cost = 0.0


    cardinality_fraction = (
        len(selected_indices)
        /
        len(all_items)
    )


    disclosure_cost = (
        weights["lambda_privacy"]
        *
        privacy
        +
        weights["lambda_cognitive"]
        *
        normalized_cognitive_cost
        +
        weights["lambda_cardinality"]
        *
        cardinality_fraction
    )


    selected_ids = tuple(
        str(
            all_items[index].item_id
        )
        for index in selected_indices
    )


    return {
        "selected_indices":
            selected_indices,

        "selected_ids":
            selected_ids,

        "selected_count":
            len(selected_indices),

        "utility":
            float(utility),

        "privacy":
            float(privacy),

        "normalized_cognitive_cost":
            float(
                normalized_cognitive_cost
            ),

        "cardinality_fraction":
            float(
                cardinality_fraction
            ),

        "disclosure_cost":
            float(disclosure_cost),
    }


# --------------------------------------------------
# Exact exhaustive solver
# --------------------------------------------------

def exact_coverage_mst(
    all_items,
    feature_weights,
    tau,
    weights,
):
    """
    Exhaustively select the minimum-disclosure-cost
    subset satisfying pool-normalized coverage utility.
    """

    start_time = time.perf_counter()

    best_result = None
    evaluated_subsets = 0


    for subset_size in range(
        len(all_items) + 1
    ):

        for selected_indices in itertools.combinations(
            range(
                len(all_items)
            ),
            subset_size,
        ):

            evaluated_subsets += 1


            result = evaluate_coverage_subset(
                all_items=all_items,
                selected_indices=
                    selected_indices,
                feature_weights=
                    feature_weights,
                weights=weights,
            )


            if (
                result["utility"]
                +
                COVERAGE_NUMERICAL_TOLERANCE
                <
                tau
            ):

                continue


            if best_result is None:

                best_result = result
                continue


            candidate_key = (
                result["disclosure_cost"],
                result["selected_count"],
                result["selected_ids"],
            )


            incumbent_key = (
                best_result[
                    "disclosure_cost"
                ],
                best_result[
                    "selected_count"
                ],
                best_result[
                    "selected_ids"
                ],
            )


            if candidate_key < incumbent_key:

                best_result = result


    runtime_seconds = (
        time.perf_counter()
        -
        start_time
    )


    if best_result is None:

        return {
            "status":
                "INFEASIBLE",

            "evaluated_subsets":
                evaluated_subsets,

            "runtime_seconds":
                runtime_seconds,
        }


    return {
        "status":
            "OPTIMAL",

        **best_result,

        "evaluated_subsets":
            evaluated_subsets,

        "runtime_seconds":
            runtime_seconds,
    }


# --------------------------------------------------
# Dynamic marginal Greedy solver
# --------------------------------------------------

def greedy_coverage_mst(
    all_items,
    feature_weights,
    tau,
    weights,
):
    """
    Dynamic Greedy selection using current marginal
    utility divided by current marginal disclosure cost.
    """

    start_time = time.perf_counter()

    selected_indices = []
    remaining_indices = set(
        range(
            len(all_items)
        )
    )


    current_result = evaluate_coverage_subset(
        all_items=all_items,
        selected_indices=
            selected_indices,
        feature_weights=
            feature_weights,
        weights=weights,
    )


    while (
        current_result["utility"]
        +
        COVERAGE_NUMERICAL_TOLERANCE
        <
        tau
        and
        remaining_indices
    ):

        candidate_rows = []


        for candidate_index in sorted(
            remaining_indices
        ):

            candidate_indices = (
                selected_indices
                +
                [
                    candidate_index
                ]
            )


            candidate_result = (
                evaluate_coverage_subset(
                    all_items=all_items,
                    selected_indices=
                        candidate_indices,
                    feature_weights=
                        feature_weights,
                    weights=weights,
                )
            )


            marginal_utility = (
                candidate_result["utility"]
                -
                current_result["utility"]
            )


            marginal_cost = (
                candidate_result[
                    "disclosure_cost"
                ]
                -
                current_result[
                    "disclosure_cost"
                ]
            )


            if marginal_cost > 0.0:

                score = (
                    marginal_utility
                    /
                    marginal_cost
                )


            elif marginal_utility > 0.0:

                score = np.inf


            else:

                score = -np.inf


            candidate_rows.append(
                {
                    "candidate_index":
                        candidate_index,

                    "candidate_id":
                        str(
                            all_items[
                                candidate_index
                            ].item_id
                        ),

                    "score":
                        float(score),

                    "marginal_utility":
                        float(
                            marginal_utility
                        ),

                    "marginal_cost":
                        float(
                            marginal_cost
                        ),

                    "result":
                        candidate_result,
                }
            )


        best_candidate = min(
            candidate_rows,
            key=lambda row: (
                -row["score"],
                -row["marginal_utility"],
                row["marginal_cost"],
                row["candidate_id"],
            ),
        )


        if (
            best_candidate[
                "marginal_utility"
            ]
            <=
            COVERAGE_NUMERICAL_TOLERANCE
        ):

            break


        selected_indices.append(
            best_candidate[
                "candidate_index"
            ]
        )


        selected_indices.sort()


        remaining_indices.remove(
            best_candidate[
                "candidate_index"
            ]
        )


        current_result = (
            best_candidate[
                "result"
            ]
        )


    runtime_seconds = (
        time.perf_counter()
        -
        start_time
    )


    if (
        current_result["utility"]
        +
        COVERAGE_NUMERICAL_TOLERANCE
        >=
        tau
    ):

        status = "FEASIBLE"

    else:

        status = "INFEASIBLE"


    return {
        "status":
            status,

        **current_result,

        "runtime_seconds":
            runtime_seconds,
    }


# --------------------------------------------------
# Generate one matched coverage group
# --------------------------------------------------

matched_instances = (
    generate_matched_coverage_instances(
        n_items=
            COVERAGE_SMOKE_N_ITEMS,

        n_features=
            COVERAGE_SMOKE_FEATURES,

        seed=
            COVERAGE_SMOKE_SEED,

        coverage_density=
            COVERAGE_SMOKE_DENSITY,

        regime_strength=
            COVERAGE_SMOKE_REGIME_STRENGTH,
    )
)


# --------------------------------------------------
# Matched-generator integrity checks
# --------------------------------------------------

reference_data = matched_instances[
    "independent"
]


reference_feature_weights = np.asarray(
    reference_data[
        "feature_weights"
    ],
    dtype=float,
)


reference_coverage_matrix = np.asarray(
    reference_data[
        "coverage_matrix"
    ],
    dtype=float,
)


reference_privacy = np.sort(
    np.asarray(
        [
            item.privacy
            for item in reference_data[
                "items"
            ]
        ],
        dtype=float,
    )
)


reference_cognitive = np.sort(
    np.asarray(
        [
            item.cognitive_cost
            for item in reference_data[
                "items"
            ]
        ],
        dtype=float,
    )
)


maximum_feature_weight_difference = 0.0
maximum_coverage_matrix_difference = 0.0
maximum_sorted_privacy_difference = 0.0
maximum_sorted_cognitive_difference = 0.0


for regime in COVERAGE_SMOKE_REGIMES:

    regime_data = matched_instances[
        regime
    ]


    maximum_feature_weight_difference = max(
        maximum_feature_weight_difference,

        float(
            np.max(
                np.abs(
                    np.asarray(
                        regime_data[
                            "feature_weights"
                        ],
                        dtype=float,
                    )
                    -
                    reference_feature_weights
                )
            )
        ),
    )


    maximum_coverage_matrix_difference = max(
        maximum_coverage_matrix_difference,

        float(
            np.max(
                np.abs(
                    np.asarray(
                        regime_data[
                            "coverage_matrix"
                        ],
                        dtype=float,
                    )
                    -
                    reference_coverage_matrix
                )
            )
        ),
    )


    regime_privacy = np.sort(
        np.asarray(
            [
                item.privacy
                for item in regime_data[
                    "items"
                ]
            ],
            dtype=float,
        )
    )


    regime_cognitive = np.sort(
        np.asarray(
            [
                item.cognitive_cost
                for item in regime_data[
                    "items"
                ]
            ],
            dtype=float,
        )
    )


    maximum_sorted_privacy_difference = max(
        maximum_sorted_privacy_difference,

        float(
            np.max(
                np.abs(
                    regime_privacy
                    -
                    reference_privacy
                )
            )
        ),
    )


    maximum_sorted_cognitive_difference = max(
        maximum_sorted_cognitive_difference,

        float(
            np.max(
                np.abs(
                    regime_cognitive
                    -
                    reference_cognitive
                )
            )
        ),
    )


assert maximum_feature_weight_difference <= 1e-15
assert maximum_coverage_matrix_difference <= 1e-15
assert maximum_sorted_privacy_difference <= 1e-15
assert maximum_sorted_cognitive_difference <= 1e-15


# --------------------------------------------------
# Run all 48 matched conditions
# --------------------------------------------------

result_rows = []


for regime in COVERAGE_SMOKE_REGIMES:

    regime_data = matched_instances[
        regime
    ]


    all_items = regime_data[
        "items"
    ]


    feature_weights = np.asarray(
        regime_data[
            "feature_weights"
        ],
        dtype=float,
    )


    full_set_utility = (
        pool_normalized_coverage_utility(
            selected_items=
                all_items,

            all_items=
                all_items,

            feature_weights=
                feature_weights,
        )
    )


    assert abs(
        full_set_utility
        -
        1.0
    ) <= COVERAGE_NUMERICAL_TOLERANCE


    for tau in COVERAGE_SMOKE_THRESHOLDS:

        for weight_profile, weights in (
            COVERAGE_SMOKE_WEIGHT_PROFILES.items()
        ):

            exact_result = exact_coverage_mst(
                all_items=all_items,
                feature_weights=
                    feature_weights,
                tau=tau,
                weights=weights,
            )


            greedy_result = greedy_coverage_mst(
                all_items=all_items,
                feature_weights=
                    feature_weights,
                tau=tau,
                weights=weights,
            )


            if (
                exact_result["status"]
                ==
                "OPTIMAL"
                and
                greedy_result["status"]
                ==
                "FEASIBLE"
            ):

                optimality_gap = (
                    greedy_result[
                        "disclosure_cost"
                    ]
                    -
                    exact_result[
                        "disclosure_cost"
                    ]
                )


                optimality_gap_percent = (
                    100.0
                    *
                    optimality_gap
                    /
                    exact_result[
                        "disclosure_cost"
                    ]
                )


                exact_set = set(
                    exact_result[
                        "selected_ids"
                    ]
                )


                greedy_set = set(
                    greedy_result[
                        "selected_ids"
                    ]
                )


                union_set = (
                    exact_set
                    |
                    greedy_set
                )


                if union_set:

                    jaccard_similarity = (
                        len(
                            exact_set
                            &
                            greedy_set
                        )
                        /
                        len(
                            union_set
                        )
                    )

                else:

                    jaccard_similarity = 1.0


                same_selected_subset = (
                    exact_set
                    ==
                    greedy_set
                )


            else:

                optimality_gap = np.nan
                optimality_gap_percent = np.nan
                jaccard_similarity = np.nan
                same_selected_subset = np.nan


            result_rows.append(
                {
                    "benchmark_version":
                        (
                            "coverage_pool_normalized_"
                            "smoke_test_v1"
                        ),

                    "matched_seed":
                        COVERAGE_SMOKE_SEED,

                    "n_items":
                        COVERAGE_SMOKE_N_ITEMS,

                    "regime":
                        regime,

                    "tau":
                        tau,

                    "weight_profile":
                        weight_profile,

                    "lambda_privacy":
                        weights[
                            "lambda_privacy"
                        ],

                    "lambda_cognitive":
                        weights[
                            "lambda_cognitive"
                        ],

                    "lambda_cardinality":
                        weights[
                            "lambda_cardinality"
                        ],

                    "full_set_utility":
                        full_set_utility,

                    "exact_status":
                        exact_result[
                            "status"
                        ],

                    "exact_selected_count":
                        exact_result.get(
                            "selected_count",
                            np.nan,
                        ),

                    "exact_selected_items":
                        "|".join(
                            exact_result.get(
                                "selected_ids",
                                (),
                            )
                        ),

                    "exact_utility":
                        exact_result.get(
                            "utility",
                            np.nan,
                        ),

                    "exact_privacy":
                        exact_result.get(
                            "privacy",
                            np.nan,
                        ),

                    "exact_normalized_cognitive_cost":
                        exact_result.get(
                            "normalized_cognitive_cost",
                            np.nan,
                        ),

                    "exact_disclosure_cost":
                        exact_result.get(
                            "disclosure_cost",
                            np.nan,
                        ),

                    "exact_runtime_seconds":
                        exact_result[
                            "runtime_seconds"
                        ],

                    "evaluated_subsets":
                        exact_result[
                            "evaluated_subsets"
                        ],

                    "greedy_status":
                        greedy_result[
                            "status"
                        ],

                    "greedy_selected_count":
                        greedy_result.get(
                            "selected_count",
                            np.nan,
                        ),

                    "greedy_selected_items":
                        "|".join(
                            greedy_result.get(
                                "selected_ids",
                                (),
                            )
                        ),

                    "greedy_utility":
                        greedy_result.get(
                            "utility",
                            np.nan,
                        ),

                    "greedy_privacy":
                        greedy_result.get(
                            "privacy",
                            np.nan,
                        ),

                    "greedy_normalized_cognitive_cost":
                        greedy_result.get(
                            "normalized_cognitive_cost",
                            np.nan,
                        ),

                    "greedy_disclosure_cost":
                        greedy_result.get(
                            "disclosure_cost",
                            np.nan,
                        ),

                    "greedy_runtime_seconds":
                        greedy_result[
                            "runtime_seconds"
                        ],

                    "optimality_gap":
                        optimality_gap,

                    "optimality_gap_percent":
                        optimality_gap_percent,

                    "same_selected_subset":
                        same_selected_subset,

                    "jaccard_similarity":
                        jaccard_similarity,
                }
            )


            print(
                "Completed coverage smoke condition:",
                regime,
                tau,
                weight_profile
            )


# --------------------------------------------------
# Results and integrity audit
# --------------------------------------------------

coverage_smoke_df = pd.DataFrame(
    result_rows
)


negative_gap_rows = int(
    (
        coverage_smoke_df[
            "optimality_gap_percent"
        ]
        <
        -1e-8
    ).sum()
)


exact_threshold_failures = int(
    (
        (
            coverage_smoke_df[
                "exact_status"
            ]
            ==
            "OPTIMAL"
        )
        &
        (
            coverage_smoke_df[
                "exact_utility"
            ]
            +
            COVERAGE_NUMERICAL_TOLERANCE
            <
            coverage_smoke_df[
                "tau"
            ]
        )
    ).sum()
)


greedy_threshold_failures = int(
    (
        (
            coverage_smoke_df[
                "greedy_status"
            ]
            ==
            "FEASIBLE"
        )
        &
        (
            coverage_smoke_df[
                "greedy_utility"
            ]
            +
            COVERAGE_NUMERICAL_TOLERANCE
            <
            coverage_smoke_df[
                "tau"
            ]
        )
    ).sum()
)


status_disagreements = int(
    (
        (
            coverage_smoke_df[
                "exact_status"
            ]
            ==
            "OPTIMAL"
        )
        &
        (
            coverage_smoke_df[
                "greedy_status"
            ]
            !=
            "FEASIBLE"
        )
    ).sum()
)


duplicate_conditions = int(
    coverage_smoke_df
    .duplicated(
        subset=[
            "regime",
            "tau",
            "weight_profile",
        ],
        keep=False,
    )
    .sum()
)


coverage_smoke_audit_df = pd.DataFrame(
    [
        {
            "rows":
                len(
                    coverage_smoke_df
                ),

            "expected_rows":
                48,

            "regimes":
                coverage_smoke_df[
                    "regime"
                ].nunique(),

            "thresholds":
                coverage_smoke_df[
                    "tau"
                ].nunique(),

            "weight_profiles":
                coverage_smoke_df[
                    "weight_profile"
                ].nunique(),

            "exact_optimal_rows":
                int(
                    (
                        coverage_smoke_df[
                            "exact_status"
                        ]
                        ==
                        "OPTIMAL"
                    ).sum()
                ),

            "greedy_feasible_rows":
                int(
                    (
                        coverage_smoke_df[
                            "greedy_status"
                        ]
                        ==
                        "FEASIBLE"
                    ).sum()
                ),

            "duplicate_conditions":
                duplicate_conditions,

            "negative_gap_rows":
                negative_gap_rows,

            "exact_threshold_failures":
                exact_threshold_failures,

            "greedy_threshold_failures":
                greedy_threshold_failures,

            "status_disagreements":
                status_disagreements,

            "maximum_feature_weight_difference":
                maximum_feature_weight_difference,

            "maximum_coverage_matrix_difference":
                maximum_coverage_matrix_difference,

            "maximum_sorted_privacy_difference":
                maximum_sorted_privacy_difference,

            "maximum_sorted_cognitive_difference":
                maximum_sorted_cognitive_difference,

            "minimum_optimality_gap_percent":
                coverage_smoke_df[
                    "optimality_gap_percent"
                ].min(),

            "maximum_optimality_gap_percent":
                coverage_smoke_df[
                    "optimality_gap_percent"
                ].max(),

            "mean_optimality_gap_percent":
                coverage_smoke_df[
                    "optimality_gap_percent"
                ].mean(),

            "exact_subset_space_size":
                int(
                    coverage_smoke_df[
                        "evaluated_subsets"
                    ].max()
                ),
        }
    ]
)


assert len(
    coverage_smoke_df
) == 48


assert duplicate_conditions == 0
assert negative_gap_rows == 0
assert exact_threshold_failures == 0
assert greedy_threshold_failures == 0
assert status_disagreements == 0


assert (
    coverage_smoke_df[
        "exact_status"
    ]
    ==
    "OPTIMAL"
).all()


assert (
    coverage_smoke_df[
        "greedy_status"
    ]
    ==
    "FEASIBLE"
).all()


assert (
    coverage_smoke_df[
        "evaluated_subsets"
    ]
    ==
    2
    **
    COVERAGE_SMOKE_N_ITEMS
).all()


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

coverage_smoke_df.to_csv(
    COVERAGE_SMOKE_RESULTS_PATH,
    index=False,
)


coverage_smoke_audit_df.to_csv(
    COVERAGE_SMOKE_AUDIT_PATH,
    index=False,
)


# --------------------------------------------------
# Printed report
# --------------------------------------------------

coverage_smoke_summary_df = (
    coverage_smoke_df
    .groupby(
        "regime",
        as_index=False,
    )
    .agg(
        conditions=(
            "optimality_gap_percent",
            "size",
        ),

        mean_optimality_gap_percent=(
            "optimality_gap_percent",
            "mean",
        ),

        median_optimality_gap_percent=(
            "optimality_gap_percent",
            "median",
        ),

        maximum_optimality_gap_percent=(
            "optimality_gap_percent",
            "max",
        ),

        exact_subset_recovery_percent=(
            "same_selected_subset",
            lambda values:
                100.0
                *
                values.astype(
                    float
                ).mean(),
        ),

        mean_jaccard_similarity=(
            "jaccard_similarity",
            "mean",
        ),

        mean_exact_selected_count=(
            "exact_selected_count",
            "mean",
        ),

        mean_greedy_selected_count=(
            "greedy_selected_count",
            "mean",
        ),

        mean_exact_runtime_seconds=(
            "exact_runtime_seconds",
            "mean",
        ),

        mean_greedy_runtime_seconds=(
            "greedy_runtime_seconds",
            "mean",
        ),
    )
)


print(
    "\n======================================"
)

print(
    "COVERAGE EXACT-VERSUS-GREEDY SMOKE AUDIT"
)

print(
    "======================================"
)


print(
    coverage_smoke_audit_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.12f}",
    )
)


print(
    "\n======================================"
)

print(
    "COVERAGE SMOKE RESULTS BY REGIME"
)

print(
    "======================================"
)


print(
    coverage_smoke_summary_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.6f}",
    )
)


print(
    "\nSmoke-test results saved to:"
)

print(
    COVERAGE_SMOKE_RESULTS_PATH
)


print(
    "\nSmoke-test audit saved to:"
)

print(
    COVERAGE_SMOKE_AUDIT_PATH
)

# ===== notebook 02, code cell 86 =====
print('### notebook cell 86', flush=True)
import os
PROJECT_ROOT = os.environ.get("MST_PROJECT_ROOT", "..")
# --------------------------------------------------
# Step 128 — Pool-normalized coverage cached-Exact
# solver validation against the brute-force smoke test
# --------------------------------------------------

from pathlib import Path
import time

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CACHED_EXACT_TOLERANCE = 1e-10


if "RESULTS_DIR" not in globals():

    RESULTS_DIR = Path(
        os.path.join(PROJECT_ROOT, "05_Results")
    )


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SMOKE_REFERENCE_PATH = (
    RESULTS_DIR
    / "coverage_exact_vs_greedy_smoke_test.csv"
)


CACHED_VALIDATION_PATH = (
    RESULTS_DIR
    / "coverage_cached_exact_smoke_validation.csv"
)


CACHED_VALIDATION_AUDIT_PATH = (
    RESULTS_DIR
    / "coverage_cached_exact_smoke_validation_audit.csv"
)


if "generate_matched_coverage_instances" not in globals():

    raise NameError(
        "generate_matched_coverage_instances is unavailable. "
        "Run Step 124 first."
    )


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def parse_pipe_subset(
    stored_subset,
):
    """
    Parse strings such as:
        c001|c004|c009
    """

    if stored_subset is None:

        return tuple()


    try:

        if pd.isna(
            stored_subset
        ):

            return tuple()

    except (
        TypeError,
        ValueError,
    ):

        pass


    text = str(
        stored_subset
    ).strip()


    if not text:

        return tuple()


    return tuple(
        token.strip()
        for token in text.split("|")
        if token.strip()
    )



def subset_ids_from_mask(
    subset_mask,
    item_ids,
):
    """
    Return selected item identifiers in increasing
    candidate-index order.
    """

    return tuple(
        item_ids[
            item_index
        ]
        for item_index in range(
            len(
                item_ids
            )
        )
        if (
            subset_mask
            >>
            item_index
        )
        &
        1
    )


# --------------------------------------------------
# Cached subset precomputation
# --------------------------------------------------

def precompute_coverage_subset_cache(
    all_items,
    feature_weights,
):
    """
    Precompute utility and disclosure components for
    all 2^M subsets.

    Utility:
        candidate-pool-normalized weighted coverage

    Privacy:
        1 - product(1 - p_i)

    Cognitive:
        sum(c_i) / sum(all candidate cognitive costs)

    Cardinality:
        |S| / M
    """

    precomputation_start = time.perf_counter()


    n_items = len(
        all_items
    )


    if n_items <= 0:

        raise ValueError(
            "The candidate pool must not be empty."
        )


    item_ids = tuple(
        str(
            item.item_id
        )
        for item in all_items
    )


    if len(
        set(
            item_ids
        )
    ) != n_items:

        raise ValueError(
            "Candidate item identifiers must be unique."
        )


    feature_weights = np.asarray(
        feature_weights,
        dtype=float,
    )


    coverage_matrix = np.asarray(
        [
            item.feature_coverage
            for item in all_items
        ],
        dtype=float,
    )


    privacy_values = np.asarray(
        [
            float(
                item.privacy
            )
            for item in all_items
        ],
        dtype=float,
    )


    cognitive_values = np.asarray(
        [
            float(
                item.cognitive_cost
            )
            for item in all_items
        ],
        dtype=float,
    )


    if coverage_matrix.ndim != 2:

        raise ValueError(
            "The coverage matrix must be two-dimensional."
        )


    if (
        coverage_matrix.shape[0]
        != n_items
    ):

        raise ValueError(
            "Coverage rows must equal the number of items."
        )


    if (
        coverage_matrix.shape[1]
        != len(
            feature_weights
        )
    ):

        raise ValueError(
            "Coverage columns must match feature weights."
        )


    if np.any(
        feature_weights < 0.0
    ):

        raise ValueError(
            "Feature weights must be nonnegative."
        )


    full_feature_coverage = (
        coverage_matrix.max(
            axis=0
        )
    )


    utility_denominator = float(
        np.dot(
            feature_weights,
            full_feature_coverage,
        )
    )


    if utility_denominator <= 0.0:

        raise ValueError(
            "The candidate pool provides no positive "
            "weighted feature coverage."
        )


    full_cognitive_cost = float(
        cognitive_values.sum()
    )


    if full_cognitive_cost <= 0.0:

        raise ValueError(
            "The candidate pool must have positive "
            "total cognitive cost."
        )


    number_of_subsets = (
        1
        <<
        n_items
    )


    number_of_features = (
        coverage_matrix.shape[1]
    )


    maximum_coverage = np.zeros(
        (
            number_of_subsets,
            number_of_features,
        ),
        dtype=float,
    )


    privacy_failure_probability = np.ones(
        number_of_subsets,
        dtype=float,
    )


    cognitive_sum = np.zeros(
        number_of_subsets,
        dtype=float,
    )


    selected_count = np.zeros(
        number_of_subsets,
        dtype=np.int16,
    )


    # Add the highest-index selected item to a subset
    # containing only lower-index items. This preserves
    # increasing item order in floating-point summation.
    for subset_mask in range(
        1,
        number_of_subsets,
    ):

        highest_item_index = (
            subset_mask.bit_length()
            -
            1
        )


        highest_item_bit = (
            1
            <<
            highest_item_index
        )


        previous_mask = (
            subset_mask
            ^
            highest_item_bit
        )


        maximum_coverage[
            subset_mask
        ] = np.maximum(
            maximum_coverage[
                previous_mask
            ],
            coverage_matrix[
                highest_item_index
            ],
        )


        privacy_failure_probability[
            subset_mask
        ] = (
            privacy_failure_probability[
                previous_mask
            ]
            *
            (
                1.0
                -
                privacy_values[
                    highest_item_index
                ]
            )
        )


        cognitive_sum[
            subset_mask
        ] = (
            cognitive_sum[
                previous_mask
            ]
            +
            cognitive_values[
                highest_item_index
            ]
        )


        selected_count[
            subset_mask
        ] = (
            selected_count[
                previous_mask
            ]
            +
            1
        )


    utility = (
        maximum_coverage
        @
        feature_weights
    ) / utility_denominator


    privacy = (
        1.0
        -
        privacy_failure_probability
    )


    normalized_cognitive_cost = (
        cognitive_sum
        /
        full_cognitive_cost
    )


    cardinality_fraction = (
        selected_count.astype(
            float
        )
        /
        n_items
    )


    # Remove negligible floating-point excursions.
    utility = np.clip(
        utility,
        0.0,
        1.0,
    )


    precomputation_runtime_seconds = (
        time.perf_counter()
        -
        precomputation_start
    )


    return {
        "n_items":
            n_items,

        "number_of_subsets":
            number_of_subsets,

        "item_ids":
            item_ids,

        "utility":
            utility,

        "privacy":
            privacy,

        "normalized_cognitive_cost":
            normalized_cognitive_cost,

        "cardinality_fraction":
            cardinality_fraction,

        "selected_count":
            selected_count,

        "precomputation_runtime_seconds":
            precomputation_runtime_seconds,

        "full_set_utility":
            float(
                utility[
                    number_of_subsets
                    -
                    1
                ]
            ),
    }


# --------------------------------------------------
# Cached Exact solve
# --------------------------------------------------

def cached_exact_coverage_mst(
    subset_cache,
    tau,
    lambda_privacy,
    lambda_cognitive,
    lambda_cardinality,
):
    """
    Find the minimum-disclosure-cost sufficient subset
    using precomputed subset quantities.
    """

    solve_start = time.perf_counter()


    utility = subset_cache[
        "utility"
    ]


    feasible_indices = np.flatnonzero(
        utility
        +
        CACHED_EXACT_TOLERANCE
        >=
        tau
    )


    if len(
        feasible_indices
    ) == 0:

        return {
            "status":
                "INFEASIBLE",

            "cached_solve_runtime_seconds":
                (
                    time.perf_counter()
                    -
                    solve_start
                ),
        }


    disclosure_cost = (
        lambda_privacy
        *
        subset_cache[
            "privacy"
        ]
        +
        lambda_cognitive
        *
        subset_cache[
            "normalized_cognitive_cost"
        ]
        +
        lambda_cardinality
        *
        subset_cache[
            "cardinality_fraction"
        ]
    )


    feasible_costs = (
        disclosure_cost[
            feasible_indices
        ]
    )


    minimum_cost = float(
        feasible_costs.min()
    )


    # Exact floating-point ties only. No tolerance is
    # used here because the brute-force reference used
    # the raw tuple cost as its primary comparison key.
    tied_indices = feasible_indices[
        feasible_costs
        ==
        minimum_cost
    ]


    if len(
        tied_indices
    ) == 1:

        best_mask = int(
            tied_indices[0]
        )


    else:

        best_mask = min(
            (
                int(
                    subset_mask
                )
                for subset_mask in tied_indices
            ),
            key=lambda subset_mask: (
                int(
                    subset_cache[
                        "selected_count"
                    ][
                        subset_mask
                    ]
                ),

                subset_ids_from_mask(
                    subset_mask=
                        subset_mask,

                    item_ids=
                        subset_cache[
                            "item_ids"
                        ],
                ),
            ),
        )


    selected_ids = subset_ids_from_mask(
        subset_mask=
            best_mask,

        item_ids=
            subset_cache[
                "item_ids"
            ],
    )


    cached_solve_runtime_seconds = (
        time.perf_counter()
        -
        solve_start
    )


    return {
        "status":
            "OPTIMAL",

        "selected_mask":
            best_mask,

        "selected_ids":
            selected_ids,

        "selected_count":
            int(
                subset_cache[
                    "selected_count"
                ][
                    best_mask
                ]
            ),

        "utility":
            float(
                subset_cache[
                    "utility"
                ][
                    best_mask
                ]
            ),

        "privacy":
            float(
                subset_cache[
                    "privacy"
                ][
                    best_mask
                ]
            ),

        "normalized_cognitive_cost":
            float(
                subset_cache[
                    "normalized_cognitive_cost"
                ][
                    best_mask
                ]
            ),

        "cardinality_fraction":
            float(
                subset_cache[
                    "cardinality_fraction"
                ][
                    best_mask
                ]
            ),

        "disclosure_cost":
            float(
                disclosure_cost[
                    best_mask
                ]
            ),

        "cached_solve_runtime_seconds":
            cached_solve_runtime_seconds,
    }


# --------------------------------------------------
# Load Step-127 brute-force reference
# --------------------------------------------------

smoke_reference_df = pd.read_csv(
    SMOKE_REFERENCE_PATH,
    low_memory=False,
)


required_reference_columns = [
    "matched_seed",
    "n_items",
    "regime",
    "tau",
    "weight_profile",
    "lambda_privacy",
    "lambda_cognitive",
    "lambda_cardinality",
    "exact_status",
    "exact_selected_count",
    "exact_selected_items",
    "exact_utility",
    "exact_privacy",
    "exact_normalized_cognitive_cost",
    "exact_disclosure_cost",
]


missing_reference_columns = [
    column_name
    for column_name in required_reference_columns
    if column_name not in smoke_reference_df.columns
]


if missing_reference_columns:

    raise KeyError(
        "Missing smoke-reference columns: "
        + ", ".join(
            missing_reference_columns
        )
    )


assert len(
    smoke_reference_df
) == 48


unique_seeds = (
    smoke_reference_df[
        "matched_seed"
    ]
    .dropna()
    .unique()
)


unique_sizes = (
    smoke_reference_df[
        "n_items"
    ]
    .dropna()
    .unique()
)


assert len(
    unique_seeds
) == 1


assert len(
    unique_sizes
) == 1


matched_seed = int(
    unique_seeds[0]
)


n_items = int(
    unique_sizes[0]
)


# --------------------------------------------------
# Recreate the identical matched group
# --------------------------------------------------

matched_instances = (
    generate_matched_coverage_instances(
        n_items=n_items,

        n_features=globals().get(
            "COVERAGE_SMOKE_FEATURES",
            globals().get(
                "COVERAGE_FEATURES",
                12,
            ),
        ),

        seed=matched_seed,

        coverage_density=globals().get(
            "COVERAGE_SMOKE_DENSITY",
            globals().get(
                "COVERAGE_DENSITY",
                0.35,
            ),
        ),

        regime_strength=globals().get(
            "COVERAGE_SMOKE_REGIME_STRENGTH",
            globals().get(
                "COVERAGE_REGIME_STRENGTH",
                0.75,
            ),
        ),
    )
)


# --------------------------------------------------
# Precompute one cache per regime
# --------------------------------------------------

subset_caches = {}


for regime in sorted(
    smoke_reference_df[
        "regime"
    ].unique()
):

    regime_data = (
        matched_instances[
            regime
        ]
    )


    subset_caches[
        regime
    ] = (
        precompute_coverage_subset_cache(
            all_items=
                regime_data[
                    "items"
                ],

            feature_weights=
                regime_data[
                    "feature_weights"
                ],
        )
    )


    print(
        "Completed cached subset precomputation:",
        regime,
        "| subsets =",
        subset_caches[
            regime
        ][
            "number_of_subsets"
        ],
        "| seconds =",
        f"{subset_caches[regime]['precomputation_runtime_seconds']:.4f}",
    )


# --------------------------------------------------
# Re-solve all 48 Exact conditions
# --------------------------------------------------

validation_rows = []


for _, reference_row in (
    smoke_reference_df
    .sort_values(
        [
            "regime",
            "tau",
            "weight_profile",
        ]
    )
    .iterrows()
):

    regime = reference_row[
        "regime"
    ]


    cached_result = (
        cached_exact_coverage_mst(
            subset_cache=
                subset_caches[
                    regime
                ],

            tau=
                float(
                    reference_row[
                        "tau"
                    ]
                ),

            lambda_privacy=
                float(
                    reference_row[
                        "lambda_privacy"
                    ]
                ),

            lambda_cognitive=
                float(
                    reference_row[
                        "lambda_cognitive"
                    ]
                ),

            lambda_cardinality=
                float(
                    reference_row[
                        "lambda_cardinality"
                    ]
                ),
        )
    )


    reference_selected_ids = (
        parse_pipe_subset(
            reference_row[
                "exact_selected_items"
            ]
        )
    )


    cached_selected_ids = tuple(
        cached_result.get(
            "selected_ids",
            tuple(),
        )
    )


    validation_rows.append(
        {
            "regime":
                regime,

            "tau":
                float(
                    reference_row[
                        "tau"
                    ]
                ),

            "weight_profile":
                reference_row[
                    "weight_profile"
                ],

            "reference_status":
                reference_row[
                    "exact_status"
                ],

            "cached_status":
                cached_result[
                    "status"
                ],

            "status_match":
                (
                    reference_row[
                        "exact_status"
                    ]
                    ==
                    cached_result[
                        "status"
                    ]
                ),

            "reference_selected_items":
                "|".join(
                    reference_selected_ids
                ),

            "cached_selected_items":
                "|".join(
                    cached_selected_ids
                ),

            "selected_subset_match":
                (
                    reference_selected_ids
                    ==
                    cached_selected_ids
                ),

            "reference_selected_count":
                int(
                    reference_row[
                        "exact_selected_count"
                    ]
                ),

            "cached_selected_count":
                cached_result.get(
                    "selected_count",
                    np.nan,
                ),

            "selected_count_match":
                (
                    int(
                        reference_row[
                            "exact_selected_count"
                        ]
                    )
                    ==
                    cached_result.get(
                        "selected_count",
                        -1,
                    )
                ),

            "absolute_utility_difference":
                abs(
                    float(
                        reference_row[
                            "exact_utility"
                        ]
                    )
                    -
                    cached_result.get(
                        "utility",
                        np.nan,
                    )
                ),

            "absolute_privacy_difference":
                abs(
                    float(
                        reference_row[
                            "exact_privacy"
                        ]
                    )
                    -
                    cached_result.get(
                        "privacy",
                        np.nan,
                    )
                ),

            "absolute_normalized_cognitive_difference":
                abs(
                    float(
                        reference_row[
                            "exact_normalized_cognitive_cost"
                        ]
                    )
                    -
                    cached_result.get(
                        "normalized_cognitive_cost",
                        np.nan,
                    )
                ),

            "absolute_disclosure_cost_difference":
                abs(
                    float(
                        reference_row[
                            "exact_disclosure_cost"
                        ]
                    )
                    -
                    cached_result.get(
                        "disclosure_cost",
                        np.nan,
                    )
                ),

            "cached_solve_runtime_seconds":
                cached_result[
                    "cached_solve_runtime_seconds"
                ],
        }
    )


validation_df = pd.DataFrame(
    validation_rows
)


# --------------------------------------------------
# Validation audit
# --------------------------------------------------

audit_df = pd.DataFrame(
    [
        {
            "validation_rows":
                len(
                    validation_df
                ),

            "expected_rows":
                48,

            "status_matches":
                int(
                    validation_df[
                        "status_match"
                    ].sum()
                ),

            "selected_subset_matches":
                int(
                    validation_df[
                        "selected_subset_match"
                    ].sum()
                ),

            "selected_count_matches":
                int(
                    validation_df[
                        "selected_count_match"
                    ].sum()
                ),

            "maximum_absolute_utility_difference":
                validation_df[
                    "absolute_utility_difference"
                ].max(),

            "maximum_absolute_privacy_difference":
                validation_df[
                    "absolute_privacy_difference"
                ].max(),

            "maximum_absolute_normalized_cognitive_difference":
                validation_df[
                    "absolute_normalized_cognitive_difference"
                ].max(),

            "maximum_absolute_disclosure_cost_difference":
                validation_df[
                    "absolute_disclosure_cost_difference"
                ].max(),

            "maximum_full_set_deviation_from_one":
                max(
                    abs(
                        subset_cache[
                            "full_set_utility"
                        ]
                        -
                        1.0
                    )
                    for subset_cache in (
                        subset_caches.values()
                    )
                ),

            "total_subset_precomputation_seconds":
                sum(
                    subset_cache[
                        "precomputation_runtime_seconds"
                    ]
                    for subset_cache in (
                        subset_caches.values()
                    )
                ),

            "mean_cached_solve_milliseconds":
                1000.0
                *
                validation_df[
                    "cached_solve_runtime_seconds"
                ].mean(),

            "maximum_cached_solve_milliseconds":
                1000.0
                *
                validation_df[
                    "cached_solve_runtime_seconds"
                ].max(),

            "subset_space_size_per_regime":
                2
                **
                n_items,
        }
    ]
)


# --------------------------------------------------
# Integrity assertions
# --------------------------------------------------

assert len(
    validation_df
) == 48


assert validation_df[
    "status_match"
].all()


assert validation_df[
    "selected_subset_match"
].all()


assert validation_df[
    "selected_count_match"
].all()


assert (
    validation_df[
        "absolute_utility_difference"
    ].max()
    <=
    CACHED_EXACT_TOLERANCE
)


assert (
    validation_df[
        "absolute_privacy_difference"
    ].max()
    <=
    CACHED_EXACT_TOLERANCE
)


assert (
    validation_df[
        "absolute_normalized_cognitive_difference"
    ].max()
    <=
    CACHED_EXACT_TOLERANCE
)


assert (
    validation_df[
        "absolute_disclosure_cost_difference"
    ].max()
    <=
    CACHED_EXACT_TOLERANCE
)


assert (
    audit_df.loc[
        0,
        "maximum_full_set_deviation_from_one",
    ]
    <=
    CACHED_EXACT_TOLERANCE
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

validation_df.to_csv(
    CACHED_VALIDATION_PATH,
    index=False,
)


audit_df.to_csv(
    CACHED_VALIDATION_AUDIT_PATH,
    index=False,
)


# --------------------------------------------------
# Printed report
# --------------------------------------------------

print(
    "\n======================================"
)

print(
    "COVERAGE CACHED-EXACT VALIDATION AUDIT"
)

print(
    "======================================"
)


print(
    audit_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.12f}",
    )
)


print(
    "\n======================================"
)

print(
    "CACHED-EXACT VALIDATION BY REGIME"
)

print(
    "======================================"
)


regime_summary_df = (
    validation_df
    .groupby(
        "regime",
        as_index=False,
    )
    .agg(
        conditions=(
            "status_match",
            "size",
        ),

        status_matches=(
            "status_match",
            "sum",
        ),

        selected_subset_matches=(
            "selected_subset_match",
            "sum",
        ),

        selected_count_matches=(
            "selected_count_match",
            "sum",
        ),

        maximum_utility_difference=(
            "absolute_utility_difference",
            "max",
        ),

        maximum_privacy_difference=(
            "absolute_privacy_difference",
            "max",
        ),

        maximum_cognitive_difference=(
            "absolute_normalized_cognitive_difference",
            "max",
        ),

        maximum_disclosure_cost_difference=(
            "absolute_disclosure_cost_difference",
            "max",
        ),

        mean_cached_solve_milliseconds=(
            "cached_solve_runtime_seconds",
            lambda values:
                1000.0
                *
                values.mean(),
        ),
    )
)


print(
    regime_summary_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.12f}",
    )
)


print(
    "\nValidation rows saved to:"
)

print(
    CACHED_VALIDATION_PATH
)


print(
    "\nValidation audit saved to:"
)

print(
    CACHED_VALIDATION_AUDIT_PATH
)

# ===== notebook 02, code cell 87 =====
print('### notebook cell 87', flush=True)
import os
PROJECT_ROOT = os.environ.get("MST_PROJECT_ROOT", "..")
# --------------------------------------------------
# Step 129 — Full pool-normalized coverage benchmark:
# Cached Exact versus Dynamic Greedy
# --------------------------------------------------

from pathlib import Path
import time

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

COVERAGE_MAIN_BENCHMARK_VERSION = (
    "coverage_pool_normalized_exact_greedy_v1"
)


COVERAGE_MAIN_BASE_SEED = (
    globals().get(
        "GLOBAL_SEED",
        2026,
    )
    + 129
)


COVERAGE_MAIN_SIZES = [
    8,
    10,
    12,
    14,
    16,
]


COVERAGE_MAIN_REPLICATES_PER_SIZE = 100


COVERAGE_MAIN_REGIMES = [
    "independent",
    "utility_burden_alignment",
    "utility_burden_conflict",
]


COVERAGE_MAIN_THRESHOLDS = [
    0.70,
    0.80,
    0.90,
    0.95,
]


COVERAGE_MAIN_WEIGHT_PROFILES = {
    "balanced": {
        "lambda_privacy": 1 / 3,
        "lambda_cognitive": 1 / 3,
        "lambda_cardinality": 1 / 3,
    },

    "privacy_priority": {
        "lambda_privacy": 0.60,
        "lambda_cognitive": 0.20,
        "lambda_cardinality": 0.20,
    },

    "interpretation_priority": {
        "lambda_privacy": 0.20,
        "lambda_cognitive": 0.60,
        "lambda_cardinality": 0.20,
    },

    "parsimony_priority": {
        "lambda_privacy": 0.20,
        "lambda_cognitive": 0.20,
        "lambda_cardinality": 0.60,
    },
}


COVERAGE_MAIN_FEATURES = globals().get(
    "COVERAGE_FEATURES",
    12,
)


COVERAGE_MAIN_DENSITY = globals().get(
    "COVERAGE_DENSITY",
    0.35,
)


COVERAGE_MAIN_REGIME_STRENGTH = globals().get(
    "COVERAGE_REGIME_STRENGTH",
    0.75,
)


COVERAGE_MAIN_TOLERANCE = 1e-12

COVERAGE_EXACT_RECOVERY_TOLERANCE_PERCENT = 1e-10

COVERAGE_MAIN_RESUME = True


EXPECTED_CONDITIONS_PER_MATCHED_GROUP = (
    len(
        COVERAGE_MAIN_REGIMES
    )
    *
    len(
        COVERAGE_MAIN_THRESHOLDS
    )
    *
    len(
        COVERAGE_MAIN_WEIGHT_PROFILES
    )
)


EXPECTED_MATCHED_GROUPS = (
    len(
        COVERAGE_MAIN_SIZES
    )
    *
    COVERAGE_MAIN_REPLICATES_PER_SIZE
)


EXPECTED_BENCHMARK_ROWS = (
    EXPECTED_MATCHED_GROUPS
    *
    EXPECTED_CONDITIONS_PER_MATCHED_GROUP
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

if "RESULTS_DIR" not in globals():

    RESULTS_DIR = Path(
        os.path.join(PROJECT_ROOT, "05_Results")
    )


if "TABLES_DIR" not in globals():

    TABLES_DIR = Path(
        os.path.join(PROJECT_ROOT, "07_Tables")
    )


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if "generate_matched_coverage_instances" not in globals():

    raise NameError(
        "generate_matched_coverage_instances is unavailable. "
        "Run Step 124 first."
    )


# --------------------------------------------------
# Output paths
# --------------------------------------------------

COVERAGE_MAIN_RESULTS_PATH = (
    RESULTS_DIR
    / (
        "coverage_exact_vs_greedy_"
        "main_benchmark_v1_pool_normalized.csv"
    )
)


COVERAGE_MAIN_GROUP_AUDIT_PATH = (
    RESULTS_DIR
    / (
        "coverage_exact_vs_greedy_"
        "main_benchmark_v1_group_audit.csv"
    )
)


COVERAGE_MAIN_OVERALL_SUMMARY_PATH = (
    TABLES_DIR
    / (
        "coverage_exact_vs_greedy_"
        "main_overall_summary.csv"
    )
)


COVERAGE_MAIN_REGIME_SUMMARY_PATH = (
    TABLES_DIR
    / (
        "coverage_exact_vs_greedy_"
        "main_by_regime_summary.csv"
    )
)


COVERAGE_MAIN_RUNTIME_SUMMARY_PATH = (
    TABLES_DIR
    / (
        "coverage_exact_vs_greedy_"
        "runtime_by_candidate_size.csv"
    )
)


COVERAGE_MAIN_METADATA_PATH = (
    RESULTS_DIR
    / (
        "coverage_exact_vs_greedy_"
        "main_benchmark_metadata.csv"
    )
)


# --------------------------------------------------
# Utility helpers
# --------------------------------------------------

def coverage_subset_ids_from_mask(
    subset_mask,
    item_ids,
):
    """
    Return selected identifiers in candidate-index
    order.
    """

    return tuple(
        item_ids[
            item_index
        ]
        for item_index in range(
            len(
                item_ids
            )
        )
        if (
            subset_mask
            >>
            item_index
        )
        &
        1
    )


def coverage_jaccard(
    first_ids,
    second_ids,
):
    """
    Jaccard similarity between two selected subsets.
    """

    first_set = set(
        first_ids
    )


    second_set = set(
        second_ids
    )


    union_set = (
        first_set
        |
        second_set
    )


    if not union_set:

        return 1.0


    return (
        len(
            first_set
            &
            second_set
        )
        /
        len(
            union_set
        )
    )


# --------------------------------------------------
# Shared utility-cache precomputation
# --------------------------------------------------

def precompute_shared_coverage_utility_cache(
    all_items,
    feature_weights,
):
    """
    Precompute pool-normalized coverage utility for all
    2^M subsets.

    The utility structure is shared across the three
    matched burden-association regimes.
    """

    start_time = time.perf_counter()


    n_items = len(
        all_items
    )


    if n_items <= 0:

        raise ValueError(
            "The candidate pool cannot be empty."
        )


    item_ids = tuple(
        str(
            item.item_id
        )
        for item in all_items
    )


    if len(
        set(
            item_ids
        )
    ) != n_items:

        raise ValueError(
            "Candidate item identifiers must be unique."
        )


    feature_weights = np.asarray(
        feature_weights,
        dtype=float,
    )


    coverage_matrix = np.asarray(
        [
            item.feature_coverage
            for item in all_items
        ],
        dtype=float,
    )


    if coverage_matrix.ndim != 2:

        raise ValueError(
            "The coverage matrix must be two-dimensional."
        )


    if (
        coverage_matrix.shape[1]
        != len(
            feature_weights
        )
    ):

        raise ValueError(
            "Coverage dimensions do not match "
            "feature weights."
        )


    available_feature_coverage = (
        coverage_matrix.max(
            axis=0
        )
    )


    utility_denominator = float(
        np.dot(
            feature_weights,
            available_feature_coverage,
        )
    )


    if utility_denominator <= 0.0:

        raise ValueError(
            "The candidate pool has zero attainable "
            "weighted coverage."
        )


    number_of_subsets = (
        1
        <<
        n_items
    )


    number_of_features = (
        coverage_matrix.shape[1]
    )


    maximum_coverage = np.zeros(
        (
            number_of_subsets,
            number_of_features,
        ),
        dtype=float,
    )


    selected_count = np.zeros(
        number_of_subsets,
        dtype=np.int16,
    )


    for subset_mask in range(
        1,
        number_of_subsets,
    ):

        highest_item_index = (
            subset_mask.bit_length()
            -
            1
        )


        highest_item_bit = (
            1
            <<
            highest_item_index
        )


        previous_mask = (
            subset_mask
            ^
            highest_item_bit
        )


        maximum_coverage[
            subset_mask
        ] = np.maximum(
            maximum_coverage[
                previous_mask
            ],
            coverage_matrix[
                highest_item_index
            ],
        )


        selected_count[
            subset_mask
        ] = (
            selected_count[
                previous_mask
            ]
            +
            1
        )


    utility = (
        maximum_coverage
        @
        feature_weights
    ) / utility_denominator


    utility = np.clip(
        utility,
        0.0,
        1.0,
    )


    cardinality_fraction = (
        selected_count.astype(
            float
        )
        /
        n_items
    )


    runtime_seconds = (
        time.perf_counter()
        -
        start_time
    )


    return {
        "n_items":
            n_items,

        "number_of_subsets":
            number_of_subsets,

        "item_ids":
            item_ids,

        "feature_weights":
            feature_weights,

        "coverage_matrix":
            coverage_matrix,

        "utility_denominator":
            utility_denominator,

        "utility":
            utility,

        "selected_count":
            selected_count,

        "cardinality_fraction":
            cardinality_fraction,

        "full_set_utility":
            float(
                utility[
                    number_of_subsets
                    -
                    1
                ]
            ),

        "runtime_seconds":
            runtime_seconds,
    }


# --------------------------------------------------
# Regime-specific burden cache
# --------------------------------------------------

def precompute_coverage_burden_cache(
    all_items,
    shared_cache,
):
    """
    Precompute privacy exposure and normalized modeled
    interpretation cost for every subset.
    """

    start_time = time.perf_counter()


    n_items = shared_cache[
        "n_items"
    ]


    number_of_subsets = shared_cache[
        "number_of_subsets"
    ]


    if len(
        all_items
    ) != n_items:

        raise ValueError(
            "Burden items do not match the shared "
            "candidate pool size."
        )


    privacy_values = np.asarray(
        [
            float(
                item.privacy
            )
            for item in all_items
        ],
        dtype=float,
    )


    cognitive_values = np.asarray(
        [
            float(
                item.cognitive_cost
            )
            for item in all_items
        ],
        dtype=float,
    )


    if np.any(
        privacy_values < 0.0
    ) or np.any(
        privacy_values > 1.0
    ):

        raise ValueError(
            "Privacy values must lie in [0, 1]."
        )


    if np.any(
        cognitive_values < 0.0
    ):

        raise ValueError(
            "Modeled interpretation costs must be "
            "nonnegative."
        )


    total_cognitive_cost = float(
        cognitive_values.sum()
    )


    if total_cognitive_cost <= 0.0:

        raise ValueError(
            "The candidate pool must have positive "
            "total modeled interpretation cost."
        )


    privacy_failure_probability = np.ones(
        number_of_subsets,
        dtype=float,
    )


    cognitive_sum = np.zeros(
        number_of_subsets,
        dtype=float,
    )


    for subset_mask in range(
        1,
        number_of_subsets,
    ):

        highest_item_index = (
            subset_mask.bit_length()
            -
            1
        )


        highest_item_bit = (
            1
            <<
            highest_item_index
        )


        previous_mask = (
            subset_mask
            ^
            highest_item_bit
        )


        privacy_failure_probability[
            subset_mask
        ] = (
            privacy_failure_probability[
                previous_mask
            ]
            *
            (
                1.0
                -
                privacy_values[
                    highest_item_index
                ]
            )
        )


        cognitive_sum[
            subset_mask
        ] = (
            cognitive_sum[
                previous_mask
            ]
            +
            cognitive_values[
                highest_item_index
            ]
        )


    privacy = (
        1.0
        -
        privacy_failure_probability
    )


    normalized_cognitive_cost = (
        cognitive_sum
        /
        total_cognitive_cost
    )


    runtime_seconds = (
        time.perf_counter()
        -
        start_time
    )


    return {
        "privacy_values":
            privacy_values,

        "cognitive_values":
            cognitive_values,

        "total_cognitive_cost":
            total_cognitive_cost,

        "privacy":
            privacy,

        "normalized_cognitive_cost":
            normalized_cognitive_cost,

        "runtime_seconds":
            runtime_seconds,
    }


# --------------------------------------------------
# Cached Exact solver
# --------------------------------------------------

def solve_cached_exact_coverage(
    shared_cache,
    burden_cache,
    tau,
    weights,
):
    """
    Select the minimum-disclosure-cost subset satisfying
    the modeled utility threshold.
    """

    start_time = time.perf_counter()


    utility = shared_cache[
        "utility"
    ]


    feasible_masks = np.flatnonzero(
        utility
        +
        COVERAGE_MAIN_TOLERANCE
        >=
        tau
    )


    if len(
        feasible_masks
    ) == 0:

        return {
            "status":
                "INFEASIBLE",

            "runtime_seconds":
                (
                    time.perf_counter()
                    -
                    start_time
                ),
        }


    disclosure_cost = (
        weights[
            "lambda_privacy"
        ]
        *
        burden_cache[
            "privacy"
        ]
        +
        weights[
            "lambda_cognitive"
        ]
        *
        burden_cache[
            "normalized_cognitive_cost"
        ]
        +
        weights[
            "lambda_cardinality"
        ]
        *
        shared_cache[
            "cardinality_fraction"
        ]
    )


    feasible_costs = (
        disclosure_cost[
            feasible_masks
        ]
    )


    minimum_cost = float(
        feasible_costs.min()
    )


    tied_masks = feasible_masks[
        feasible_costs
        ==
        minimum_cost
    ]


    if len(
        tied_masks
    ) == 1:

        best_mask = int(
            tied_masks[0]
        )


    else:

        best_mask = min(
            (
                int(
                    subset_mask
                )
                for subset_mask in tied_masks
            ),
            key=lambda subset_mask: (
                int(
                    shared_cache[
                        "selected_count"
                    ][
                        subset_mask
                    ]
                ),

                coverage_subset_ids_from_mask(
                    subset_mask,
                    shared_cache[
                        "item_ids"
                    ],
                ),
            ),
        )


    selected_ids = coverage_subset_ids_from_mask(
        best_mask,
        shared_cache[
            "item_ids"
        ],
    )


    return {
        "status":
            "OPTIMAL",

        "selected_mask":
            best_mask,

        "selected_ids":
            selected_ids,

        "selected_count":
            int(
                shared_cache[
                    "selected_count"
                ][
                    best_mask
                ]
            ),

        "utility":
            float(
                shared_cache[
                    "utility"
                ][
                    best_mask
                ]
            ),

        "privacy":
            float(
                burden_cache[
                    "privacy"
                ][
                    best_mask
                ]
            ),

        "normalized_cognitive_cost":
            float(
                burden_cache[
                    "normalized_cognitive_cost"
                ][
                    best_mask
                ]
            ),

        "cardinality_fraction":
            float(
                shared_cache[
                    "cardinality_fraction"
                ][
                    best_mask
                ]
            ),

        "disclosure_cost":
            float(
                disclosure_cost[
                    best_mask
                ]
            ),

        "runtime_seconds":
            (
                time.perf_counter()
                -
                start_time
            ),
    }


# --------------------------------------------------
# Dynamic marginal Greedy solver
# --------------------------------------------------

def solve_dynamic_greedy_coverage(
    all_items,
    shared_cache,
    burden_cache,
    tau,
    weights,
):
    """
    Dynamic Greedy selection based on current marginal
    modeled utility divided by current marginal
    disclosure cost.
    """

    start_time = time.perf_counter()


    n_items = shared_cache[
        "n_items"
    ]


    feature_weights = shared_cache[
        "feature_weights"
    ]


    coverage_matrix = shared_cache[
        "coverage_matrix"
    ]


    utility_denominator = shared_cache[
        "utility_denominator"
    ]


    privacy_values = burden_cache[
        "privacy_values"
    ]


    cognitive_values = burden_cache[
        "cognitive_values"
    ]


    total_cognitive_cost = burden_cache[
        "total_cognitive_cost"
    ]


    selected_indices = []
    remaining_indices = set(
        range(
            n_items
        )
    )


    current_feature_coverage = np.zeros(
        coverage_matrix.shape[1],
        dtype=float,
    )


    current_utility = 0.0
    current_privacy = 0.0
    current_normalized_cognitive = 0.0
    current_cardinality_fraction = 0.0


    current_disclosure_cost = 0.0


    while (
        current_utility
        +
        COVERAGE_MAIN_TOLERANCE
        <
        tau
        and
        remaining_indices
    ):

        candidate_rows = []


        for candidate_index in sorted(
            remaining_indices
        ):

            candidate_feature_coverage = np.maximum(
                current_feature_coverage,
                coverage_matrix[
                    candidate_index
                ],
            )


            candidate_utility = float(
                np.dot(
                    feature_weights,
                    candidate_feature_coverage,
                )
                /
                utility_denominator
            )


            candidate_privacy = (
                1.0
                -
                (
                    1.0
                    -
                    current_privacy
                )
                *
                (
                    1.0
                    -
                    privacy_values[
                        candidate_index
                    ]
                )
            )


            candidate_normalized_cognitive = (
                current_normalized_cognitive
                +
                cognitive_values[
                    candidate_index
                ]
                /
                total_cognitive_cost
            )


            candidate_cardinality_fraction = (
                current_cardinality_fraction
                +
                1.0
                /
                n_items
            )


            candidate_disclosure_cost = (
                weights[
                    "lambda_privacy"
                ]
                *
                candidate_privacy
                +
                weights[
                    "lambda_cognitive"
                ]
                *
                candidate_normalized_cognitive
                +
                weights[
                    "lambda_cardinality"
                ]
                *
                candidate_cardinality_fraction
            )


            marginal_utility = (
                candidate_utility
                -
                current_utility
            )


            marginal_cost = (
                candidate_disclosure_cost
                -
                current_disclosure_cost
            )


            if marginal_cost > 0.0:

                score = (
                    marginal_utility
                    /
                    marginal_cost
                )


            elif marginal_utility > 0.0:

                score = np.inf


            else:

                score = -np.inf


            candidate_rows.append(
                {
                    "candidate_index":
                        candidate_index,

                    "candidate_id":
                        str(
                            all_items[
                                candidate_index
                            ].item_id
                        ),

                    "candidate_feature_coverage":
                        candidate_feature_coverage,

                    "candidate_utility":
                        candidate_utility,

                    "candidate_privacy":
                        candidate_privacy,

                    "candidate_normalized_cognitive":
                        candidate_normalized_cognitive,

                    "candidate_cardinality_fraction":
                        candidate_cardinality_fraction,

                    "candidate_disclosure_cost":
                        candidate_disclosure_cost,

                    "marginal_utility":
                        marginal_utility,

                    "marginal_cost":
                        marginal_cost,

                    "score":
                        score,
                }
            )


        best_candidate = min(
            candidate_rows,
            key=lambda row: (
                -row[
                    "score"
                ],

                -row[
                    "marginal_utility"
                ],

                row[
                    "marginal_cost"
                ],

                row[
                    "candidate_id"
                ],
            ),
        )


        if (
            best_candidate[
                "marginal_utility"
            ]
            <=
            COVERAGE_MAIN_TOLERANCE
        ):

            break


        selected_indices.append(
            best_candidate[
                "candidate_index"
            ]
        )


        selected_indices.sort()


        remaining_indices.remove(
            best_candidate[
                "candidate_index"
            ]
        )


        current_feature_coverage = (
            best_candidate[
                "candidate_feature_coverage"
            ]
        )


        current_utility = (
            best_candidate[
                "candidate_utility"
            ]
        )


        current_privacy = (
            best_candidate[
                "candidate_privacy"
            ]
        )


        current_normalized_cognitive = (
            best_candidate[
                "candidate_normalized_cognitive"
            ]
        )


        current_cardinality_fraction = (
            best_candidate[
                "candidate_cardinality_fraction"
            ]
        )


        current_disclosure_cost = (
            best_candidate[
                "candidate_disclosure_cost"
            ]
        )


    selected_ids = tuple(
        str(
            all_items[
                item_index
            ].item_id
        )
        for item_index in selected_indices
    )


    status = (
        "FEASIBLE"
        if (
            current_utility
            +
            COVERAGE_MAIN_TOLERANCE
            >=
            tau
        )
        else
        "INFEASIBLE"
    )


    return {
        "status":
            status,

        "selected_ids":
            selected_ids,

        "selected_count":
            len(
                selected_ids
            ),

        "utility":
            float(
                current_utility
            ),

        "privacy":
            float(
                current_privacy
            ),

        "normalized_cognitive_cost":
            float(
                current_normalized_cognitive
            ),

        "cardinality_fraction":
            float(
                current_cardinality_fraction
            ),

        "disclosure_cost":
            float(
                current_disclosure_cost
            ),

        "runtime_seconds":
            (
                time.perf_counter()
                -
                start_time
            ),
    }


# --------------------------------------------------
# Resume completed checkpoint groups
# --------------------------------------------------

if (
    COVERAGE_MAIN_RESUME
    and
    COVERAGE_MAIN_RESULTS_PATH.exists()
):

    existing_results_df = pd.read_csv(
        COVERAGE_MAIN_RESULTS_PATH,
        low_memory=False,
    )


    if (
        "benchmark_version"
        in existing_results_df.columns
    ):

        existing_results_df = (
            existing_results_df[
                existing_results_df[
                    "benchmark_version"
                ]
                ==
                COVERAGE_MAIN_BENCHMARK_VERSION
            ]
            .copy()
        )


    existing_group_counts = (
        existing_results_df
        .groupby(
            "matched_group_id"
        )
        .size()
    )


    completed_group_ids = set(
        existing_group_counts[
            existing_group_counts
            ==
            EXPECTED_CONDITIONS_PER_MATCHED_GROUP
        ].index
    )


    existing_results_df = (
        existing_results_df[
            existing_results_df[
                "matched_group_id"
            ].isin(
                completed_group_ids
            )
        ]
        .copy()
    )


else:

    existing_results_df = pd.DataFrame()

    completed_group_ids = set()


if (
    COVERAGE_MAIN_RESUME
    and
    COVERAGE_MAIN_GROUP_AUDIT_PATH.exists()
):

    existing_audit_df = pd.read_csv(
        COVERAGE_MAIN_GROUP_AUDIT_PATH,
        low_memory=False,
    )


    existing_audit_df = (
        existing_audit_df[
            existing_audit_df[
                "matched_group_id"
            ].isin(
                completed_group_ids
            )
        ]
        .copy()
    )


else:

    existing_audit_df = pd.DataFrame()


print(
    "Previously completed matched groups:",
    len(
        completed_group_ids
    )
)


# --------------------------------------------------
# Run benchmark
# --------------------------------------------------

new_result_rows = []
new_audit_rows = []

benchmark_start_time = time.perf_counter()


for n_items in COVERAGE_MAIN_SIZES:

    for replicate in range(
        COVERAGE_MAIN_REPLICATES_PER_SIZE
    ):

        matched_group_id = (
            f"COVB_M{n_items}_"
            f"R{replicate:03d}"
        )


        if matched_group_id in completed_group_ids:

            continue


        matched_seed = (
            83_000_000
            +
            100_000
            *
            n_items
            +
            COVERAGE_MAIN_BASE_SEED
            +
            replicate
        )


        matched_instances = (
            generate_matched_coverage_instances(
                n_items=
                    n_items,

                n_features=
                    COVERAGE_MAIN_FEATURES,

                seed=
                    matched_seed,

                coverage_density=
                    COVERAGE_MAIN_DENSITY,

                regime_strength=
                    COVERAGE_MAIN_REGIME_STRENGTH,
            )
        )


        reference_data = (
            matched_instances[
                "independent"
            ]
        )


        shared_cache = (
            precompute_shared_coverage_utility_cache(
                all_items=
                    reference_data[
                        "items"
                    ],

                feature_weights=
                    reference_data[
                        "feature_weights"
                    ],
            )
        )


        # ------------------------------------------
        # Matched-generator audit
        # ------------------------------------------

        reference_feature_weights = np.asarray(
            reference_data[
                "feature_weights"
            ],
            dtype=float,
        )


        reference_coverage_matrix = np.asarray(
            reference_data[
                "coverage_matrix"
            ],
            dtype=float,
        )


        reference_privacy = np.sort(
            np.asarray(
                [
                    item.privacy
                    for item in reference_data[
                        "items"
                    ]
                ],
                dtype=float,
            )
        )


        reference_cognitive = np.sort(
            np.asarray(
                [
                    item.cognitive_cost
                    for item in reference_data[
                        "items"
                    ]
                ],
                dtype=float,
            )
        )


        maximum_feature_weight_difference = 0.0
        maximum_coverage_matrix_difference = 0.0
        maximum_sorted_privacy_difference = 0.0
        maximum_sorted_cognitive_difference = 0.0


        for regime in COVERAGE_MAIN_REGIMES:

            regime_data = (
                matched_instances[
                    regime
                ]
            )


            maximum_feature_weight_difference = max(
                maximum_feature_weight_difference,

                float(
                    np.max(
                        np.abs(
                            np.asarray(
                                regime_data[
                                    "feature_weights"
                                ],
                                dtype=float,
                            )
                            -
                            reference_feature_weights
                        )
                    )
                ),
            )


            maximum_coverage_matrix_difference = max(
                maximum_coverage_matrix_difference,

                float(
                    np.max(
                        np.abs(
                            np.asarray(
                                regime_data[
                                    "coverage_matrix"
                                ],
                                dtype=float,
                            )
                            -
                            reference_coverage_matrix
                        )
                    )
                ),
            )


            regime_privacy = np.sort(
                np.asarray(
                    [
                        item.privacy
                        for item in regime_data[
                            "items"
                        ]
                    ],
                    dtype=float,
                )
            )


            regime_cognitive = np.sort(
                np.asarray(
                    [
                        item.cognitive_cost
                        for item in regime_data[
                            "items"
                        ]
                    ],
                    dtype=float,
                )
            )


            maximum_sorted_privacy_difference = max(
                maximum_sorted_privacy_difference,

                float(
                    np.max(
                        np.abs(
                            regime_privacy
                            -
                            reference_privacy
                        )
                    )
                ),
            )


            maximum_sorted_cognitive_difference = max(
                maximum_sorted_cognitive_difference,

                float(
                    np.max(
                        np.abs(
                            regime_cognitive
                            -
                            reference_cognitive
                        )
                    )
                ),
            )


        new_audit_rows.append(
            {
                "benchmark_version":
                    COVERAGE_MAIN_BENCHMARK_VERSION,

                "matched_group_id":
                    matched_group_id,

                "n_items":
                    n_items,

                "replicate":
                    replicate,

                "matched_seed":
                    matched_seed,

                "number_of_subsets":
                    shared_cache[
                        "number_of_subsets"
                    ],

                "full_set_utility":
                    shared_cache[
                        "full_set_utility"
                    ],

                "shared_utility_precomputation_seconds":
                    shared_cache[
                        "runtime_seconds"
                    ],

                "maximum_feature_weight_difference":
                    maximum_feature_weight_difference,

                "maximum_coverage_matrix_difference":
                    maximum_coverage_matrix_difference,

                "maximum_sorted_privacy_difference":
                    maximum_sorted_privacy_difference,

                "maximum_sorted_cognitive_difference":
                    maximum_sorted_cognitive_difference,
            }
        )


        for regime in COVERAGE_MAIN_REGIMES:

            regime_data = (
                matched_instances[
                    regime
                ]
            )


            all_items = (
                regime_data[
                    "items"
                ]
            )


            burden_cache = (
                precompute_coverage_burden_cache(
                    all_items=
                        all_items,

                    shared_cache=
                        shared_cache,
                )
            )


            for tau in COVERAGE_MAIN_THRESHOLDS:

                for (
                    weight_profile,
                    weights,
                ) in (
                    COVERAGE_MAIN_WEIGHT_PROFILES.items()
                ):

                    exact_result = (
                        solve_cached_exact_coverage(
                            shared_cache=
                                shared_cache,

                            burden_cache=
                                burden_cache,

                            tau=
                                tau,

                            weights=
                                weights,
                        )
                    )


                    greedy_result = (
                        solve_dynamic_greedy_coverage(
                            all_items=
                                all_items,

                            shared_cache=
                                shared_cache,

                            burden_cache=
                                burden_cache,

                            tau=
                                tau,

                            weights=
                                weights,
                        )
                    )


                    if (
                        exact_result[
                            "status"
                        ]
                        ==
                        "OPTIMAL"
                        and
                        greedy_result[
                            "status"
                        ]
                        ==
                        "FEASIBLE"
                    ):

                        optimality_gap = (
                            greedy_result[
                                "disclosure_cost"
                            ]
                            -
                            exact_result[
                                "disclosure_cost"
                            ]
                        )


                        if (
                            abs(
                                optimality_gap
                            )
                            <=
                            COVERAGE_MAIN_TOLERANCE
                        ):

                            optimality_gap = 0.0


                        optimality_gap_percent = (
                            100.0
                            *
                            optimality_gap
                            /
                            exact_result[
                                "disclosure_cost"
                            ]
                        )


                        exact_ids = (
                            exact_result[
                                "selected_ids"
                            ]
                        )


                        greedy_ids = (
                            greedy_result[
                                "selected_ids"
                            ]
                        )


                        same_selected_subset = (
                            set(
                                exact_ids
                            )
                            ==
                            set(
                                greedy_ids
                            )
                        )


                        jaccard_similarity = (
                            coverage_jaccard(
                                exact_ids,
                                greedy_ids,
                            )
                        )


                        selected_count_difference = (
                            greedy_result[
                                "selected_count"
                            ]
                            -
                            exact_result[
                                "selected_count"
                            ]
                        )


                    else:

                        optimality_gap = np.nan
                        optimality_gap_percent = np.nan
                        same_selected_subset = np.nan
                        jaccard_similarity = np.nan
                        selected_count_difference = np.nan


                    new_result_rows.append(
                        {
                            "benchmark_version":
                                COVERAGE_MAIN_BENCHMARK_VERSION,

                            "utility_family":
                                (
                                    "pool_normalized_"
                                    "weighted_maximum_coverage"
                                ),

                            "matched_group_id":
                                matched_group_id,

                            "instance_id":
                                (
                                    f"{matched_group_id}_"
                                    f"{regime}"
                                ),

                            "n_items":
                                n_items,

                            "replicate":
                                replicate,

                            "matched_seed":
                                matched_seed,

                            "regime":
                                regime,

                            "tau":
                                tau,

                            "weight_profile":
                                weight_profile,

                            "lambda_privacy":
                                weights[
                                    "lambda_privacy"
                                ],

                            "lambda_cognitive":
                                weights[
                                    "lambda_cognitive"
                                ],

                            "lambda_cardinality":
                                weights[
                                    "lambda_cardinality"
                                ],

                            "full_set_utility":
                                shared_cache[
                                    "full_set_utility"
                                ],

                            "number_of_subsets":
                                shared_cache[
                                    "number_of_subsets"
                                ],

                            "shared_utility_precomputation_seconds":
                                shared_cache[
                                    "runtime_seconds"
                                ],

                            "burden_precomputation_seconds":
                                burden_cache[
                                    "runtime_seconds"
                                ],

                            "exact_status":
                                exact_result[
                                    "status"
                                ],

                            "exact_selected_count":
                                exact_result.get(
                                    "selected_count",
                                    np.nan,
                                ),

                            "exact_selected_items":
                                "|".join(
                                    exact_result.get(
                                        "selected_ids",
                                        (),
                                    )
                                ),

                            "exact_utility":
                                exact_result.get(
                                    "utility",
                                    np.nan,
                                ),

                            "exact_privacy":
                                exact_result.get(
                                    "privacy",
                                    np.nan,
                                ),

                            "exact_normalized_cognitive_cost":
                                exact_result.get(
                                    "normalized_cognitive_cost",
                                    np.nan,
                                ),

                            "exact_cardinality_fraction":
                                exact_result.get(
                                    "cardinality_fraction",
                                    np.nan,
                                ),

                            "exact_disclosure_cost":
                                exact_result.get(
                                    "disclosure_cost",
                                    np.nan,
                                ),

                            "exact_runtime_seconds":
                                exact_result[
                                    "runtime_seconds"
                                ],

                            "greedy_status":
                                greedy_result[
                                    "status"
                                ],

                            "greedy_selected_count":
                                greedy_result.get(
                                    "selected_count",
                                    np.nan,
                                ),

                            "greedy_selected_items":
                                "|".join(
                                    greedy_result.get(
                                        "selected_ids",
                                        (),
                                    )
                                ),

                            "greedy_utility":
                                greedy_result.get(
                                    "utility",
                                    np.nan,
                                ),

                            "greedy_privacy":
                                greedy_result.get(
                                    "privacy",
                                    np.nan,
                                ),

                            "greedy_normalized_cognitive_cost":
                                greedy_result.get(
                                    "normalized_cognitive_cost",
                                    np.nan,
                                ),

                            "greedy_cardinality_fraction":
                                greedy_result.get(
                                    "cardinality_fraction",
                                    np.nan,
                                ),

                            "greedy_disclosure_cost":
                                greedy_result.get(
                                    "disclosure_cost",
                                    np.nan,
                                ),

                            "greedy_runtime_seconds":
                                greedy_result[
                                    "runtime_seconds"
                                ],

                            "optimality_gap":
                                optimality_gap,

                            "optimality_gap_percent":
                                optimality_gap_percent,

                            "same_selected_subset":
                                same_selected_subset,

                            "jaccard_similarity":
                                jaccard_similarity,

                            "selected_count_difference":
                                selected_count_difference,
                        }
                    )


        if (
            (
                replicate
                +
                1
            )
            %
            10
            ==
            0
        ):

            print(
                "Completed coverage benchmark:",
                f"M={n_items}",
                f"replicate={replicate + 1}/"
                f"{COVERAGE_MAIN_REPLICATES_PER_SIZE}",
            )


    # ----------------------------------------------
    # Size-level checkpoint
    # ----------------------------------------------

    new_results_df = pd.DataFrame(
        new_result_rows
    )


    checkpoint_results_df = pd.concat(
        [
            existing_results_df,
            new_results_df,
        ],
        ignore_index=True,
    )


    checkpoint_results_df = (
        checkpoint_results_df
        .drop_duplicates(
            subset=[
                "matched_group_id",
                "regime",
                "tau",
                "weight_profile",
            ],
            keep="last",
        )
        .sort_values(
            [
                "n_items",
                "replicate",
                "regime",
                "tau",
                "weight_profile",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    checkpoint_results_df.to_csv(
        COVERAGE_MAIN_RESULTS_PATH,
        index=False,
    )


    new_audit_df = pd.DataFrame(
        new_audit_rows
    )


    checkpoint_audit_df = pd.concat(
        [
            existing_audit_df,
            new_audit_df,
        ],
        ignore_index=True,
    )


    checkpoint_audit_df = (
        checkpoint_audit_df
        .drop_duplicates(
            subset=[
                "matched_group_id",
            ],
            keep="last",
        )
        .sort_values(
            [
                "n_items",
                "replicate",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    checkpoint_audit_df.to_csv(
        COVERAGE_MAIN_GROUP_AUDIT_PATH,
        index=False,
    )


    print(
        "Checkpoint saved after candidate size:",
        n_items,
        "| current rows:",
        len(
            checkpoint_results_df
        ),
    )


# --------------------------------------------------
# Final benchmark data
# --------------------------------------------------

coverage_main_df = pd.read_csv(
    COVERAGE_MAIN_RESULTS_PATH,
    low_memory=False,
)


coverage_group_audit_df = pd.read_csv(
    COVERAGE_MAIN_GROUP_AUDIT_PATH,
    low_memory=False,
)


for column_name in [
    "n_items",
    "tau",
    "exact_selected_count",
    "greedy_selected_count",
    "exact_utility",
    "greedy_utility",
    "exact_disclosure_cost",
    "greedy_disclosure_cost",
    "optimality_gap_percent",
    "jaccard_similarity",
    "selected_count_difference",
]:

    coverage_main_df[
        column_name
    ] = pd.to_numeric(
        coverage_main_df[
            column_name
        ],
        errors="coerce",
    )


# --------------------------------------------------
# Reporting indicators
# --------------------------------------------------

coverage_main_df[
    "exact_cost_recovery_numeric"
] = (
    coverage_main_df[
        "optimality_gap_percent"
    ]
    .abs()
    <=
    COVERAGE_EXACT_RECOVERY_TOLERANCE_PERCENT
).astype(int)


coverage_main_df[
    "same_selected_subset_numeric"
] = (
    coverage_main_df[
        "same_selected_subset"
    ]
    .astype(str)
    .str.lower()
    .eq(
        "true"
    )
    .astype(int)
)


coverage_main_df[
    "large_gap_numeric"
] = (
    coverage_main_df[
        "optimality_gap_percent"
    ]
    >
    10.0
).astype(int)


coverage_main_df[
    "more_items_numeric"
] = (
    coverage_main_df[
        "selected_count_difference"
    ]
    >
    0
).astype(int)


coverage_main_df[
    "within_1_percent_numeric"
] = (
    coverage_main_df[
        "optimality_gap_percent"
    ]
    <=
    1.0
).astype(int)


coverage_main_df[
    "within_5_percent_numeric"
] = (
    coverage_main_df[
        "optimality_gap_percent"
    ]
    <=
    5.0
).astype(int)


coverage_main_df[
    "within_10_percent_numeric"
] = (
    coverage_main_df[
        "optimality_gap_percent"
    ]
    <=
    10.0
).astype(int)


# --------------------------------------------------
# Integrity audit
# --------------------------------------------------

duplicate_conditions = int(
    coverage_main_df
    .duplicated(
        subset=[
            "matched_group_id",
            "regime",
            "tau",
            "weight_profile",
        ],
        keep=False,
    )
    .sum()
)


negative_gap_rows = int(
    (
        coverage_main_df[
            "optimality_gap_percent"
        ]
        <
        -1e-8
    ).sum()
)


exact_threshold_failures = int(
    (
        (
            coverage_main_df[
                "exact_status"
            ]
            ==
            "OPTIMAL"
        )
        &
        (
            coverage_main_df[
                "exact_utility"
            ]
            +
            COVERAGE_MAIN_TOLERANCE
            <
            coverage_main_df[
                "tau"
            ]
        )
    ).sum()
)


greedy_threshold_failures = int(
    (
        (
            coverage_main_df[
                "greedy_status"
            ]
            ==
            "FEASIBLE"
        )
        &
        (
            coverage_main_df[
                "greedy_utility"
            ]
            +
            COVERAGE_MAIN_TOLERANCE
            <
            coverage_main_df[
                "tau"
            ]
        )
    ).sum()
)


status_disagreements = int(
    (
        (
            coverage_main_df[
                "exact_status"
            ]
            ==
            "OPTIMAL"
        )
        &
        (
            coverage_main_df[
                "greedy_status"
            ]
            !=
            "FEASIBLE"
        )
    ).sum()
)


group_condition_counts = (
    coverage_main_df
    .groupby(
        "matched_group_id"
    )
    .size()
)


coverage_integrity_audit_df = pd.DataFrame(
    [
        {
            "benchmark_rows":
                len(
                    coverage_main_df
                ),

            "expected_benchmark_rows":
                EXPECTED_BENCHMARK_ROWS,

            "matched_groups":
                coverage_main_df[
                    "matched_group_id"
                ].nunique(),

            "expected_matched_groups":
                EXPECTED_MATCHED_GROUPS,

            "minimum_conditions_per_group":
                int(
                    group_condition_counts.min()
                ),

            "maximum_conditions_per_group":
                int(
                    group_condition_counts.max()
                ),

            "exact_optimal_rows":
                int(
                    (
                        coverage_main_df[
                            "exact_status"
                        ]
                        ==
                        "OPTIMAL"
                    ).sum()
                ),

            "greedy_feasible_rows":
                int(
                    (
                        coverage_main_df[
                            "greedy_status"
                        ]
                        ==
                        "FEASIBLE"
                    ).sum()
                ),

            "duplicate_conditions":
                duplicate_conditions,

            "negative_gap_rows":
                negative_gap_rows,

            "exact_threshold_failures":
                exact_threshold_failures,

            "greedy_threshold_failures":
                greedy_threshold_failures,

            "status_disagreements":
                status_disagreements,

            "maximum_full_set_deviation_from_one":
                (
                    coverage_group_audit_df[
                        "full_set_utility"
                    ]
                    -
                    1.0
                )
                .abs()
                .max(),

            "maximum_feature_weight_difference":
                coverage_group_audit_df[
                    "maximum_feature_weight_difference"
                ].max(),

            "maximum_coverage_matrix_difference":
                coverage_group_audit_df[
                    "maximum_coverage_matrix_difference"
                ].max(),

            "maximum_sorted_privacy_difference":
                coverage_group_audit_df[
                    "maximum_sorted_privacy_difference"
                ].max(),

            "maximum_sorted_cognitive_difference":
                coverage_group_audit_df[
                    "maximum_sorted_cognitive_difference"
                ].max(),

            "minimum_optimality_gap_percent":
                coverage_main_df[
                    "optimality_gap_percent"
                ].min(),

            "maximum_optimality_gap_percent":
                coverage_main_df[
                    "optimality_gap_percent"
                ].max(),

            "elapsed_seconds_this_run":
                (
                    time.perf_counter()
                    -
                    benchmark_start_time
                ),
        }
    ]
)


assert len(
    coverage_main_df
) == EXPECTED_BENCHMARK_ROWS


assert (
    coverage_main_df[
        "matched_group_id"
    ].nunique()
    ==
    EXPECTED_MATCHED_GROUPS
)


assert (
    group_condition_counts
    ==
    EXPECTED_CONDITIONS_PER_MATCHED_GROUP
).all()


assert duplicate_conditions == 0
assert negative_gap_rows == 0
assert exact_threshold_failures == 0
assert greedy_threshold_failures == 0
assert status_disagreements == 0


assert (
    coverage_main_df[
        "exact_status"
    ]
    ==
    "OPTIMAL"
).all()


assert (
    coverage_main_df[
        "greedy_status"
    ]
    ==
    "FEASIBLE"
).all()


assert (
    coverage_integrity_audit_df.loc[
        0,
        "maximum_full_set_deviation_from_one",
    ]
    <=
    COVERAGE_MAIN_TOLERANCE
)


assert (
    coverage_integrity_audit_df.loc[
        0,
        "maximum_feature_weight_difference",
    ]
    <=
    COVERAGE_MAIN_TOLERANCE
)


assert (
    coverage_integrity_audit_df.loc[
        0,
        "maximum_coverage_matrix_difference",
    ]
    <=
    COVERAGE_MAIN_TOLERANCE
)


assert (
    coverage_integrity_audit_df.loc[
        0,
        "maximum_sorted_privacy_difference",
    ]
    <=
    COVERAGE_MAIN_TOLERANCE
)


assert (
    coverage_integrity_audit_df.loc[
        0,
        "maximum_sorted_cognitive_difference",
    ]
    <=
    COVERAGE_MAIN_TOLERANCE
)


# --------------------------------------------------
# Summary helper
# --------------------------------------------------

def summarize_coverage_algorithm_quality(
    data_frame,
):
    """
    Descriptive Exact-relative Dynamic Greedy quality.
    """

    gap_values = (
        data_frame[
            "optimality_gap_percent"
        ]
        .to_numpy(
            dtype=float
        )
    )


    return pd.Series(
        {
            "comparable_rows":
                len(
                    data_frame
                ),

            "mean_optimality_gap_percent":
                gap_values.mean(),

            "median_optimality_gap_percent":
                np.median(
                    gap_values
                ),

            "p90_optimality_gap_percent":
                np.quantile(
                    gap_values,
                    0.90,
                ),

            "p95_optimality_gap_percent":
                np.quantile(
                    gap_values,
                    0.95,
                ),

            "maximum_optimality_gap_percent":
                gap_values.max(),

            "exact_cost_recovery_percent":
                100.0
                *
                data_frame[
                    "exact_cost_recovery_numeric"
                ].mean(),

            "exact_subset_recovery_percent":
                100.0
                *
                data_frame[
                    "same_selected_subset_numeric"
                ].mean(),

            "mean_jaccard_similarity":
                data_frame[
                    "jaccard_similarity"
                ].mean(),

            "large_gap_percent":
                100.0
                *
                data_frame[
                    "large_gap_numeric"
                ].mean(),

            "more_items_percent":
                100.0
                *
                data_frame[
                    "more_items_numeric"
                ].mean(),

            "mean_selected_count_difference":
                data_frame[
                    "selected_count_difference"
                ].mean(),

            "within_1_percent":
                100.0
                *
                data_frame[
                    "within_1_percent_numeric"
                ].mean(),

            "within_5_percent":
                100.0
                *
                data_frame[
                    "within_5_percent_numeric"
                ].mean(),

            "within_10_percent":
                100.0
                *
                data_frame[
                    "within_10_percent_numeric"
                ].mean(),

            "mean_exact_selected_count":
                data_frame[
                    "exact_selected_count"
                ].mean(),

            "mean_greedy_selected_count":
                data_frame[
                    "greedy_selected_count"
                ].mean(),
        }
    )


# --------------------------------------------------
# Overall and regime summaries
# --------------------------------------------------

coverage_overall_summary_df = (
    summarize_coverage_algorithm_quality(
        coverage_main_df
    )
    .to_frame()
    .T
)


coverage_overall_summary_df.insert(
    0,
    "utility_family",
    (
        "pool_normalized_"
        "weighted_maximum_coverage"
    ),
)


coverage_regime_summary_df = (
    coverage_main_df
    .groupby(
        "regime",
        sort=False,
    )
    .apply(
        summarize_coverage_algorithm_quality,
        include_groups=False,
    )
    .reset_index()
)


# --------------------------------------------------
# Runtime summary by candidate-set size
# --------------------------------------------------

group_runtime_df = (
    coverage_group_audit_df[
        [
            "matched_group_id",
            "n_items",
            "shared_utility_precomputation_seconds",
        ]
    ]
    .drop_duplicates(
        "matched_group_id"
    )
)


burden_runtime_df = (
    coverage_main_df[
        [
            "matched_group_id",
            "n_items",
            "regime",
            "burden_precomputation_seconds",
        ]
    ]
    .drop_duplicates(
        [
            "matched_group_id",
            "regime",
        ]
    )
)


utility_runtime_summary_df = (
    group_runtime_df
    .groupby(
        "n_items",
        as_index=False,
    )
    .agg(
        matched_groups=(
            "matched_group_id",
            "nunique",
        ),

        mean_shared_utility_precomputation_seconds=(
            "shared_utility_precomputation_seconds",
            "mean",
        ),

        median_shared_utility_precomputation_seconds=(
            "shared_utility_precomputation_seconds",
            "median",
        ),
    )
)


burden_runtime_summary_df = (
    burden_runtime_df
    .groupby(
        "n_items",
        as_index=False,
    )
    .agg(
        mean_burden_precomputation_seconds=(
            "burden_precomputation_seconds",
            "mean",
        ),

        median_burden_precomputation_seconds=(
            "burden_precomputation_seconds",
            "median",
        ),
    )
)


solver_runtime_summary_df = (
    coverage_main_df
    .groupby(
        "n_items",
        as_index=False,
    )
    .agg(
        conditions=(
            "optimality_gap_percent",
            "size",
        ),

        mean_exact_solve_milliseconds=(
            "exact_runtime_seconds",
            lambda values:
                1000.0
                *
                values.mean(),
        ),

        median_exact_solve_milliseconds=(
            "exact_runtime_seconds",
            lambda values:
                1000.0
                *
                values.median(),
        ),

        mean_greedy_solve_milliseconds=(
            "greedy_runtime_seconds",
            lambda values:
                1000.0
                *
                values.mean(),
        ),

        median_greedy_solve_milliseconds=(
            "greedy_runtime_seconds",
            lambda values:
                1000.0
                *
                values.median(),
        ),
    )
)


coverage_runtime_summary_df = (
    utility_runtime_summary_df
    .merge(
        burden_runtime_summary_df,
        on="n_items",
        how="left",
    )
    .merge(
        solver_runtime_summary_df,
        on="n_items",
        how="left",
    )
)


# --------------------------------------------------
# Metadata
# --------------------------------------------------

coverage_metadata_df = pd.DataFrame(
    [
        {
            "benchmark_version":
                COVERAGE_MAIN_BENCHMARK_VERSION,

            "utility_family":
                (
                    "pool-normalized weighted "
                    "maximum coverage"
                ),

            "base_seed":
                COVERAGE_MAIN_BASE_SEED,

            "candidate_sizes":
                repr(
                    COVERAGE_MAIN_SIZES
                ),

            "replicates_per_size":
                COVERAGE_MAIN_REPLICATES_PER_SIZE,

            "total_matched_groups":
                EXPECTED_MATCHED_GROUPS,

            "regimes":
                repr(
                    COVERAGE_MAIN_REGIMES
                ),

            "thresholds":
                repr(
                    COVERAGE_MAIN_THRESHOLDS
                ),

            "weight_profiles":
                repr(
                    list(
                        COVERAGE_MAIN_WEIGHT_PROFILES.keys()
                    )
                ),

            "total_conditions":
                EXPECTED_BENCHMARK_ROWS,

            "n_features":
                COVERAGE_MAIN_FEATURES,

            "coverage_density":
                COVERAGE_MAIN_DENSITY,

            "regime_strength":
                COVERAGE_MAIN_REGIME_STRENGTH,

            "exact_method":
                (
                    "complete subset enumeration "
                    "with cached utility and burden arrays"
                ),

            "heuristic":
                (
                    "dynamic marginal utility divided "
                    "by dynamic marginal disclosure cost"
                ),

            "threshold_interpretation":
                (
                    "fraction of available modeled "
                    "decision-support utility"
                ),

            "approximation_ratio_claimed":
                False,
        }
    ]
)


# --------------------------------------------------
# Save final tables
# --------------------------------------------------

coverage_overall_summary_df.to_csv(
    COVERAGE_MAIN_OVERALL_SUMMARY_PATH,
    index=False,
)


coverage_regime_summary_df.to_csv(
    COVERAGE_MAIN_REGIME_SUMMARY_PATH,
    index=False,
)


coverage_runtime_summary_df.to_csv(
    COVERAGE_MAIN_RUNTIME_SUMMARY_PATH,
    index=False,
)


coverage_metadata_df.to_csv(
    COVERAGE_MAIN_METADATA_PATH,
    index=False,
)


# --------------------------------------------------
# Printed report
# --------------------------------------------------

print(
    "\n======================================"
)

print(
    "COVERAGE MAIN BENCHMARK INTEGRITY AUDIT"
)

print(
    "======================================"
)


print(
    coverage_integrity_audit_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.12f}",
    )
)


print(
    "\n======================================"
)

print(
    "COVERAGE EXACT-VERSUS-GREEDY OVERALL RESULTS"
)

print(
    "======================================"
)


print(
    coverage_overall_summary_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


print(
    "\n======================================"
)

print(
    "COVERAGE EXACT-VERSUS-GREEDY RESULTS BY REGIME"
)

print(
    "======================================"
)


print(
    coverage_regime_summary_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


print(
    "\n======================================"
)

print(
    "COVERAGE BENCHMARK RUNTIME BY CANDIDATE SIZE"
)

print(
    "======================================"
)


print(
    coverage_runtime_summary_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.6f}",
    )
)


print(
    "\nMain benchmark saved to:"
)

print(
    COVERAGE_MAIN_RESULTS_PATH
)


print(
    "\nGroup-level integrity audit saved to:"
)

print(
    COVERAGE_MAIN_GROUP_AUDIT_PATH
)


print(
    "\nOverall summary saved to:"
)

print(
    COVERAGE_MAIN_OVERALL_SUMMARY_PATH
)


print(
    "\nRegime summary saved to:"
)

print(
    COVERAGE_MAIN_REGIME_SUMMARY_PATH
)


print(
    "\nRuntime summary saved to:"
)

print(
    COVERAGE_MAIN_RUNTIME_SUMMARY_PATH
)

# ===== notebook 02, code cell 88 =====
print('### notebook cell 88', flush=True)
import os
PROJECT_ROOT = os.environ.get("MST_PROJECT_ROOT", "..")
# --------------------------------------------------
# Step 130 — Coverage benchmark paired structural-
# regime contrasts with matched-group bootstrap
# --------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

COVERAGE_REGIME_BOOTSTRAP_REPLICATES = 5000

COVERAGE_REGIME_BOOTSTRAP_SEED = (
    globals().get(
        "GLOBAL_SEED",
        2026,
    )
    + 130
)


if "RESULTS_DIR" not in globals():

    RESULTS_DIR = Path(
        os.path.join(PROJECT_ROOT, "05_Results")
    )


if "TABLES_DIR" not in globals():

    TABLES_DIR = Path(
        os.path.join(PROJECT_ROOT, "07_Tables")
    )


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


COVERAGE_MAIN_RESULTS_PATH = (
    RESULTS_DIR
    / (
        "coverage_exact_vs_greedy_"
        "main_benchmark_v1_pool_normalized.csv"
    )
)


COVERAGE_REGIME_DESCRIPTIVE_PATH = (
    TABLES_DIR
    / "coverage_exact_vs_greedy_regime_descriptive_summary.csv"
)


COVERAGE_REGIME_CONTRAST_PATH = (
    TABLES_DIR
    / "coverage_exact_vs_greedy_paired_regime_contrasts.csv"
)


COVERAGE_REGIME_REPLICATES_PATH = (
    RESULTS_DIR
    / "coverage_exact_vs_greedy_paired_regime_bootstrap_replicates.csv"
)


COVERAGE_REGIME_AUDIT_PATH = (
    RESULTS_DIR
    / "coverage_exact_vs_greedy_regime_contrast_audit.csv"
)


REGIME_ORDER = [
    "independent",
    "utility_burden_alignment",
    "utility_burden_conflict",
]


CANDIDATE_SIZE_ORDER = [
    8,
    10,
    12,
    14,
    16,
]


REGIME_CONTRASTS = [
    (
        "utility_burden_alignment",
        "independent",
    ),

    (
        "utility_burden_conflict",
        "independent",
    ),

    (
        "utility_burden_conflict",
        "utility_burden_alignment",
    ),
]


METRIC_NAMES = [
    "mean_optimality_gap_percent",
    "exact_cost_recovery_percent",
    "mean_jaccard_similarity",
    "large_gap_percent",
    "more_items_percent",
    "mean_selected_count_difference",
    "within_1_percent",
    "within_5_percent",
    "within_10_percent",
]


AGGREGATE_COLUMNS = [
    "comparable_rows",
    "gap_sum",
    "exact_cost_recovery_sum",
    "jaccard_sum",
    "large_gap_sum",
    "more_items_sum",
    "selected_count_difference_sum",
    "within_1_percent_sum",
    "within_5_percent_sum",
    "within_10_percent_sum",
]


EXACT_COST_RECOVERY_TOLERANCE_PERCENT = 1e-10


# --------------------------------------------------
# Load coverage benchmark
# --------------------------------------------------

coverage_df = pd.read_csv(
    COVERAGE_MAIN_RESULTS_PATH,
    low_memory=False,
)


required_columns = [
    "matched_group_id",
    "n_items",
    "regime",
    "tau",
    "weight_profile",
    "exact_status",
    "greedy_status",
    "optimality_gap_percent",
    "jaccard_similarity",
    "selected_count_difference",
]


missing_columns = [
    column_name
    for column_name in required_columns
    if column_name not in coverage_df.columns
]


if missing_columns:

    raise KeyError(
        "Missing required columns: "
        + ", ".join(
            missing_columns
        )
    )


for column_name in [
    "n_items",
    "tau",
    "optimality_gap_percent",
    "jaccard_similarity",
    "selected_count_difference",
]:

    coverage_df[
        column_name
    ] = pd.to_numeric(
        coverage_df[
            column_name
        ],
        errors="coerce",
    )


# --------------------------------------------------
# Benchmark design validation
# --------------------------------------------------

assert len(
    coverage_df
) == 24000


assert (
    coverage_df[
        "matched_group_id"
    ].nunique()
    == 500
)


assert set(
    coverage_df[
        "regime"
    ].dropna().unique()
) == set(
    REGIME_ORDER
)


assert (
    sorted(
        coverage_df[
            "n_items"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    ==
    CANDIDATE_SIZE_ORDER
)


assert (
    coverage_df[
        "exact_status"
    ]
    ==
    "OPTIMAL"
).all()


assert (
    coverage_df[
        "greedy_status"
    ]
    ==
    "FEASIBLE"
).all()


# --------------------------------------------------
# Reporting indicators
# --------------------------------------------------

coverage_df[
    "exact_cost_recovery_numeric"
] = (
    coverage_df[
        "optimality_gap_percent"
    ]
    .abs()
    <=
    EXACT_COST_RECOVERY_TOLERANCE_PERCENT
).astype(int)


coverage_df[
    "large_gap_numeric"
] = (
    coverage_df[
        "optimality_gap_percent"
    ]
    >
    10.0
).astype(int)


coverage_df[
    "more_items_numeric"
] = (
    coverage_df[
        "selected_count_difference"
    ]
    >
    0
).astype(int)


coverage_df[
    "within_1_percent_numeric"
] = (
    coverage_df[
        "optimality_gap_percent"
    ]
    <=
    1.0
).astype(int)


coverage_df[
    "within_5_percent_numeric"
] = (
    coverage_df[
        "optimality_gap_percent"
    ]
    <=
    5.0
).astype(int)


coverage_df[
    "within_10_percent_numeric"
] = (
    coverage_df[
        "optimality_gap_percent"
    ]
    <=
    10.0
).astype(int)


# --------------------------------------------------
# Descriptive regime summary
# --------------------------------------------------

descriptive_rows = []


for regime in REGIME_ORDER:

    current_df = (
        coverage_df[
            coverage_df[
                "regime"
            ]
            ==
            regime
        ]
    )


    gap_values = (
        current_df[
            "optimality_gap_percent"
        ]
        .to_numpy(
            dtype=float
        )
    )


    descriptive_rows.append(
        {
            "regime":
                regime,

            "comparable_rows":
                len(
                    current_df
                ),

            "matched_groups":
                current_df[
                    "matched_group_id"
                ].nunique(),

            "mean_optimality_gap_percent":
                gap_values.mean(),

            "median_optimality_gap_percent":
                np.median(
                    gap_values
                ),

            "p90_optimality_gap_percent":
                np.quantile(
                    gap_values,
                    0.90,
                ),

            "p95_optimality_gap_percent":
                np.quantile(
                    gap_values,
                    0.95,
                ),

            "maximum_optimality_gap_percent":
                gap_values.max(),

            "exact_cost_recovery_percent":
                100.0
                *
                current_df[
                    "exact_cost_recovery_numeric"
                ].mean(),

            "mean_jaccard_similarity":
                current_df[
                    "jaccard_similarity"
                ].mean(),

            "large_gap_percent":
                100.0
                *
                current_df[
                    "large_gap_numeric"
                ].mean(),

            "more_items_percent":
                100.0
                *
                current_df[
                    "more_items_numeric"
                ].mean(),

            "mean_selected_count_difference":
                current_df[
                    "selected_count_difference"
                ].mean(),

            "within_1_percent":
                100.0
                *
                current_df[
                    "within_1_percent_numeric"
                ].mean(),

            "within_5_percent":
                100.0
                *
                current_df[
                    "within_5_percent_numeric"
                ].mean(),

            "within_10_percent":
                100.0
                *
                current_df[
                    "within_10_percent_numeric"
                ].mean(),
        }
    )


coverage_regime_descriptive_df = pd.DataFrame(
    descriptive_rows
)


coverage_regime_descriptive_df.to_csv(
    COVERAGE_REGIME_DESCRIPTIVE_PATH,
    index=False,
)


# --------------------------------------------------
# Matched-group sufficient statistics
# --------------------------------------------------

cluster_df = (
    coverage_df
    .groupby(
        [
            "n_items",
            "matched_group_id",
            "regime",
        ],
        as_index=False,
    )
    .agg(
        comparable_rows=(
            "optimality_gap_percent",
            "size",
        ),

        gap_sum=(
            "optimality_gap_percent",
            "sum",
        ),

        exact_cost_recovery_sum=(
            "exact_cost_recovery_numeric",
            "sum",
        ),

        jaccard_sum=(
            "jaccard_similarity",
            "sum",
        ),

        large_gap_sum=(
            "large_gap_numeric",
            "sum",
        ),

        more_items_sum=(
            "more_items_numeric",
            "sum",
        ),

        selected_count_difference_sum=(
            "selected_count_difference",
            "sum",
        ),

        within_1_percent_sum=(
            "within_1_percent_numeric",
            "sum",
        ),

        within_5_percent_sum=(
            "within_5_percent_numeric",
            "sum",
        ),

        within_10_percent_sum=(
            "within_10_percent_numeric",
            "sum",
        ),
    )
)


# Each matched group contains:
# 4 thresholds × 4 weight profiles = 16 rows/regime.
assert (
    cluster_df[
        "comparable_rows"
    ]
    ==
    16
).all()


# --------------------------------------------------
# Metric conversion
# --------------------------------------------------

def calculate_regime_metrics(
    totals,
):
    """
    Convert sufficient-statistic totals into
    Exact-relative Dynamic Greedy metrics.
    """

    comparable_rows = totals[0]


    if comparable_rows <= 0:

        return np.full(
            len(
                METRIC_NAMES
            ),
            np.nan,
        )


    return np.array(
        [
            totals[1]
            /
            comparable_rows,

            100.0
            *
            totals[2]
            /
            comparable_rows,

            totals[3]
            /
            comparable_rows,

            100.0
            *
            totals[4]
            /
            comparable_rows,

            100.0
            *
            totals[5]
            /
            comparable_rows,

            totals[6]
            /
            comparable_rows,

            100.0
            *
            totals[7]
            /
            comparable_rows,

            100.0
            *
            totals[8]
            /
            comparable_rows,

            100.0
            *
            totals[9]
            /
            comparable_rows,
        ],
        dtype=float,
    )


# --------------------------------------------------
# Construct aligned arrays by size and regime
# --------------------------------------------------

arrays_by_size_regime = {}

support_rows = []


for n_items in CANDIDATE_SIZE_ORDER:

    arrays_by_size_regime[
        n_items
    ] = {}


    reference_group_ids = None


    for regime in REGIME_ORDER:

        current_df = (
            cluster_df[
                (
                    cluster_df[
                        "n_items"
                    ]
                    ==
                    n_items
                )
                &
                (
                    cluster_df[
                        "regime"
                    ]
                    ==
                    regime
                )
            ]
            .sort_values(
                "matched_group_id"
            )
            .reset_index(
                drop=True
            )
        )


        current_group_ids = (
            current_df[
                "matched_group_id"
            ].tolist()
        )


        assert len(
            current_group_ids
        ) == 100


        if reference_group_ids is None:

            reference_group_ids = (
                current_group_ids
            )

        else:

            assert (
                current_group_ids
                ==
                reference_group_ids
            )


        arrays_by_size_regime[
            n_items
        ][
            regime
        ] = (
            current_df[
                AGGREGATE_COLUMNS
            ]
            .to_numpy(
                dtype=float
            )
        )


        support_rows.append(
            {
                "n_items":
                    n_items,

                "regime":
                    regime,

                "matched_groups":
                    len(
                        current_df
                    ),

                "conditions":
                    int(
                        current_df[
                            "comparable_rows"
                        ].sum()
                    ),

                "minimum_conditions_per_group":
                    int(
                        current_df[
                            "comparable_rows"
                        ].min()
                    ),

                "maximum_conditions_per_group":
                    int(
                        current_df[
                            "comparable_rows"
                        ].max()
                    ),
            }
        )


support_df = pd.DataFrame(
    support_rows
)


# --------------------------------------------------
# Observed regime metrics
# --------------------------------------------------

observed_totals = {
    regime: np.zeros(
        len(
            AGGREGATE_COLUMNS
        ),
        dtype=float,
    )
    for regime in REGIME_ORDER
}


for n_items in CANDIDATE_SIZE_ORDER:

    for regime in REGIME_ORDER:

        observed_totals[
            regime
        ] += (
            arrays_by_size_regime[
                n_items
            ][
                regime
            ]
            .sum(
                axis=0
            )
        )


observed_metrics = {
    regime:
        calculate_regime_metrics(
            observed_totals[
                regime
            ]
        )
    for regime in REGIME_ORDER
}


# --------------------------------------------------
# Shared matched-group bootstrap
#
# The same resampled group indices are used across
# all three regimes within each candidate-set size.
# --------------------------------------------------

bootstrap_rng = np.random.default_rng(
    COVERAGE_REGIME_BOOTSTRAP_SEED
)


number_of_contrasts = len(
    REGIME_CONTRASTS
)


number_of_metrics = len(
    METRIC_NAMES
)


bootstrap_differences = np.empty(
    (
        COVERAGE_REGIME_BOOTSTRAP_REPLICATES,
        number_of_contrasts,
        number_of_metrics,
    ),
    dtype=float,
)


for bootstrap_index in range(
    COVERAGE_REGIME_BOOTSTRAP_REPLICATES
):

    bootstrap_totals = {
        regime: np.zeros(
            len(
                AGGREGATE_COLUMNS
            ),
            dtype=float,
        )
        for regime in REGIME_ORDER
    }


    for n_items in CANDIDATE_SIZE_ORDER:

        number_of_groups = (
            arrays_by_size_regime[
                n_items
            ][
                REGIME_ORDER[0]
            ].shape[0]
        )


        sampled_indices = (
            bootstrap_rng.integers(
                low=0,
                high=number_of_groups,
                size=number_of_groups,
            )
        )


        for regime in REGIME_ORDER:

            bootstrap_totals[
                regime
            ] += (
                arrays_by_size_regime[
                    n_items
                ][
                    regime
                ][
                    sampled_indices
                ]
                .sum(
                    axis=0
                )
            )


    bootstrap_metrics = {
        regime:
            calculate_regime_metrics(
                bootstrap_totals[
                    regime
                ]
            )
        for regime in REGIME_ORDER
    }


    for contrast_index, (
        first_regime,
        second_regime,
    ) in enumerate(
        REGIME_CONTRASTS
    ):

        bootstrap_differences[
            bootstrap_index,
            contrast_index,
        ] = (
            bootstrap_metrics[
                first_regime
            ]
            -
            bootstrap_metrics[
                second_regime
            ]
        )


# --------------------------------------------------
# Paired contrast confidence intervals
# --------------------------------------------------

contrast_rows = []


for contrast_index, (
    first_regime,
    second_regime,
) in enumerate(
    REGIME_CONTRASTS
):

    observed_difference = (
        observed_metrics[
            first_regime
        ]
        -
        observed_metrics[
            second_regime
        ]
    )


    current_bootstrap = (
        bootstrap_differences[
            :,
            contrast_index,
            :,
        ]
    )


    standard_errors = (
        current_bootstrap.std(
            axis=0,
            ddof=1,
        )
    )


    confidence_lower = np.quantile(
        current_bootstrap,
        0.025,
        axis=0,
    )


    confidence_upper = np.quantile(
        current_bootstrap,
        0.975,
        axis=0,
    )


    for metric_index, metric_name in enumerate(
        METRIC_NAMES
    ):

        contrast_rows.append(
            {
                "first_regime":
                    first_regime,

                "second_regime":
                    second_regime,

                "contrast":
                    (
                        f"{first_regime}"
                        f" minus "
                        f"{second_regime}"
                    ),

                "metric":
                    metric_name,

                "estimated_difference":
                    observed_difference[
                        metric_index
                    ],

                "bootstrap_standard_error":
                    standard_errors[
                        metric_index
                    ],

                "ci_95_lower":
                    confidence_lower[
                        metric_index
                    ],

                "ci_95_upper":
                    confidence_upper[
                        metric_index
                    ],

                "ci_excludes_zero":
                    (
                        confidence_lower[
                            metric_index
                        ]
                        >
                        0.0
                    )
                    or
                    (
                        confidence_upper[
                            metric_index
                        ]
                        <
                        0.0
                    ),

                "matched_groups":
                    500,

                "matched_groups_per_size":
                    100,

                "candidate_size_strata":
                    len(
                        CANDIDATE_SIZE_ORDER
                    ),

                "bootstrap_replicates":
                    COVERAGE_REGIME_BOOTSTRAP_REPLICATES,

                "resampling_unit":
                    "matched_size_replicate_group",

                "resampling_design":
                    (
                        "shared_across_regimes_"
                        "stratified_by_candidate_set_size"
                    ),

                "multiple_comparison_adjustment":
                    "none",
            }
        )


coverage_regime_contrast_df = pd.DataFrame(
    contrast_rows
)


# --------------------------------------------------
# Save bootstrap replicates
# --------------------------------------------------

replicate_rows = []


for bootstrap_index in range(
    COVERAGE_REGIME_BOOTSTRAP_REPLICATES
):

    for contrast_index, (
        first_regime,
        second_regime,
    ) in enumerate(
        REGIME_CONTRASTS
    ):

        replicate_row = {
            "bootstrap_replicate":
                bootstrap_index,

            "first_regime":
                first_regime,

            "second_regime":
                second_regime,

            "contrast":
                (
                    f"{first_regime}"
                    f" minus "
                    f"{second_regime}"
                ),
        }


        for metric_index, metric_name in enumerate(
            METRIC_NAMES
        ):

            replicate_row[
                metric_name
            ] = (
                bootstrap_differences[
                    bootstrap_index,
                    contrast_index,
                    metric_index,
                ]
            )


        replicate_rows.append(
            replicate_row
        )


coverage_regime_replicates_df = pd.DataFrame(
    replicate_rows
)


# --------------------------------------------------
# Audit
# --------------------------------------------------

group_condition_counts = (
    coverage_df
    .groupby(
        [
            "matched_group_id",
            "regime",
        ]
    )
    .size()
)


coverage_regime_audit_df = pd.DataFrame(
    [
        {
            "benchmark_rows":
                len(
                    coverage_df
                ),

            "matched_groups":
                coverage_df[
                    "matched_group_id"
                ].nunique(),

            "regimes":
                coverage_df[
                    "regime"
                ].nunique(),

            "candidate_sizes":
                coverage_df[
                    "n_items"
                ].nunique(),

            "minimum_conditions_per_group_regime":
                int(
                    group_condition_counts.min()
                ),

            "maximum_conditions_per_group_regime":
                int(
                    group_condition_counts.max()
                ),

            "negative_gap_rows":
                int(
                    (
                        coverage_df[
                            "optimality_gap_percent"
                        ]
                        <
                        -1e-8
                    ).sum()
                ),

            "exact_nonoptimal_rows":
                int(
                    (
                        coverage_df[
                            "exact_status"
                        ]
                        !=
                        "OPTIMAL"
                    ).sum()
                ),

            "greedy_infeasible_rows":
                int(
                    (
                        coverage_df[
                            "greedy_status"
                        ]
                        !=
                        "FEASIBLE"
                    ).sum()
                ),

            "bootstrap_replicates":
                COVERAGE_REGIME_BOOTSTRAP_REPLICATES,

            "contrasts":
                len(
                    REGIME_CONTRASTS
                ),

            "confidence_intervals":
                "unadjusted_percentile_95",
        }
    ]
)


assert (
    group_condition_counts
    ==
    16
).all()


assert coverage_regime_audit_df.loc[
    0,
    "negative_gap_rows",
] == 0


assert coverage_regime_audit_df.loc[
    0,
    "exact_nonoptimal_rows",
] == 0


assert coverage_regime_audit_df.loc[
    0,
    "greedy_infeasible_rows",
] == 0


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

coverage_regime_contrast_df.to_csv(
    COVERAGE_REGIME_CONTRAST_PATH,
    index=False,
)


coverage_regime_replicates_df.to_csv(
    COVERAGE_REGIME_REPLICATES_PATH,
    index=False,
)


coverage_regime_audit_df.to_csv(
    COVERAGE_REGIME_AUDIT_PATH,
    index=False,
)


# --------------------------------------------------
# Readable paired-contrast display
# --------------------------------------------------

key_metrics = [
    "mean_optimality_gap_percent",
    "exact_cost_recovery_percent",
    "mean_jaccard_similarity",
    "large_gap_percent",
    "more_items_percent",
    "mean_selected_count_difference",
]


contrast_display_df = (
    coverage_regime_contrast_df[
        coverage_regime_contrast_df[
            "metric"
        ].isin(
            key_metrics
        )
    ]
    .copy()
)


contrast_display_df[
    "difference_and_ci"
] = (
    contrast_display_df[
        "estimated_difference"
    ].map(
        lambda value:
            f"{value:.4f}"
    )
    +
    " ["
    +
    contrast_display_df[
        "ci_95_lower"
    ].map(
        lambda value:
            f"{value:.4f}"
    )
    +
    ", "
    +
    contrast_display_df[
        "ci_95_upper"
    ].map(
        lambda value:
            f"{value:.4f}"
    )
    +
    "]"
)


# --------------------------------------------------
# Printed report
# --------------------------------------------------

print(
    "======================================"
)

print(
    "COVERAGE REGIME-CONTRAST AUDIT"
)

print(
    "======================================"
)


print(
    coverage_regime_audit_df.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "COVERAGE RESULTS BY STRUCTURAL REGIME"
)

print(
    "======================================"
)


print(
    coverage_regime_descriptive_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


print(
    "\n======================================"
)

print(
    "PAIRED COVERAGE REGIME CONTRASTS"
)

print(
    "======================================"
)


print(
    contrast_display_df[
        [
            "first_regime",
            "second_regime",
            "metric",
            "difference_and_ci",
            "ci_excludes_zero",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nRegime descriptive summary saved to:"
)

print(
    COVERAGE_REGIME_DESCRIPTIVE_PATH
)


print(
    "\nPaired regime contrasts saved to:"
)

print(
    COVERAGE_REGIME_CONTRAST_PATH
)


print(
    "\nBootstrap replicates saved to:"
)

print(
    COVERAGE_REGIME_REPLICATES_PATH
)


print(
    "\nRegime-contrast audit saved to:"
)

print(
    COVERAGE_REGIME_AUDIT_PATH
)

# ===== notebook 02, code cell 89 =====
print('### notebook cell 89', flush=True)
import os
PROJECT_ROOT = os.environ.get("MST_PROJECT_ROOT", "..")
# --------------------------------------------------
# Step 131 — Coverage Dynamic Greedy failure
# mechanisms and extreme-tail audit
# --------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------

if "RESULTS_DIR" not in globals():

    RESULTS_DIR = Path(
        os.path.join(PROJECT_ROOT, "05_Results")
    )


if "TABLES_DIR" not in globals():

    TABLES_DIR = Path(
        os.path.join(PROJECT_ROOT, "07_Tables")
    )


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


COVERAGE_MAIN_RESULTS_PATH = (
    RESULTS_DIR
    / (
        "coverage_exact_vs_greedy_"
        "main_benchmark_v1_pool_normalized.csv"
    )
)


COVERAGE_MECHANISM_SUMMARY_PATH = (
    TABLES_DIR
    / "coverage_greedy_failure_mechanism_summary.csv"
)


COVERAGE_MECHANISM_BY_REGIME_PATH = (
    TABLES_DIR
    / "coverage_greedy_failure_mechanism_by_regime.csv"
)


COVERAGE_LARGE_GAP_ATTRIBUTION_PATH = (
    TABLES_DIR
    / "coverage_large_gap_mechanism_attribution.csv"
)


COVERAGE_TAIL_SUMMARY_PATH = (
    TABLES_DIR
    / "coverage_optimality_gap_tail_summary.csv"
)


COVERAGE_EXTREME_ROWS_PATH = (
    RESULTS_DIR
    / "coverage_extreme_optimality_gap_rows.csv"
)


COVERAGE_MECHANISM_AUDIT_PATH = (
    RESULTS_DIR
    / "coverage_failure_mechanism_audit.csv"
)


# --------------------------------------------------
# Load benchmark
# --------------------------------------------------

coverage_df = pd.read_csv(
    COVERAGE_MAIN_RESULTS_PATH,
    low_memory=False,
)


required_columns = [
    "matched_group_id",
    "n_items",
    "regime",
    "tau",
    "weight_profile",
    "exact_status",
    "greedy_status",
    "exact_selected_count",
    "greedy_selected_count",
    "exact_selected_items",
    "greedy_selected_items",
    "exact_utility",
    "greedy_utility",
    "exact_privacy",
    "greedy_privacy",
    "exact_normalized_cognitive_cost",
    "greedy_normalized_cognitive_cost",
    "exact_disclosure_cost",
    "greedy_disclosure_cost",
    "optimality_gap",
    "optimality_gap_percent",
    "jaccard_similarity",
    "selected_count_difference",
]


missing_columns = [
    column_name
    for column_name in required_columns
    if column_name not in coverage_df.columns
]


if missing_columns:

    raise KeyError(
        "Missing required columns: "
        + ", ".join(
            missing_columns
        )
    )


numeric_columns = [
    "n_items",
    "tau",
    "exact_selected_count",
    "greedy_selected_count",
    "exact_utility",
    "greedy_utility",
    "exact_privacy",
    "greedy_privacy",
    "exact_normalized_cognitive_cost",
    "greedy_normalized_cognitive_cost",
    "exact_disclosure_cost",
    "greedy_disclosure_cost",
    "optimality_gap",
    "optimality_gap_percent",
    "jaccard_similarity",
    "selected_count_difference",
]


for column_name in numeric_columns:

    coverage_df[
        column_name
    ] = pd.to_numeric(
        coverage_df[
            column_name
        ],
        errors="coerce",
    )


# --------------------------------------------------
# Integrity checks
# --------------------------------------------------

assert len(
    coverage_df
) == 24000


assert (
    coverage_df[
        "exact_status"
    ]
    ==
    "OPTIMAL"
).all()


assert (
    coverage_df[
        "greedy_status"
    ]
    ==
    "FEASIBLE"
).all()


assert (
    coverage_df[
        "optimality_gap_percent"
    ]
    >=
    -1e-8
).all()


# --------------------------------------------------
# Parse selected subsets
# --------------------------------------------------

def parse_selected_subset(
    stored_value,
):
    """
    Parse a pipe-delimited selected subset.
    """

    if stored_value is None:

        return frozenset()


    try:

        if pd.isna(
            stored_value
        ):

            return frozenset()

    except (
        TypeError,
        ValueError,
    ):

        pass


    text = str(
        stored_value
    ).strip()


    if not text:

        return frozenset()


    return frozenset(
        token.strip()
        for token in text.split("|")
        if token.strip()
    )


coverage_df[
    "exact_subset"
] = coverage_df[
    "exact_selected_items"
].map(
    parse_selected_subset
)


coverage_df[
    "greedy_subset"
] = coverage_df[
    "greedy_selected_items"
].map(
    parse_selected_subset
)


coverage_df[
    "subsets_identical"
] = (
    coverage_df[
        "exact_subset"
    ]
    ==
    coverage_df[
        "greedy_subset"
    ]
)


# --------------------------------------------------
# Failure-mechanism classification
# --------------------------------------------------

def classify_failure_mechanism(
    row,
):
    """
    Partition each condition into one mutually
    exclusive mechanism.
    """

    if row[
        "subsets_identical"
    ]:

        return "identical_subset"


    count_difference = row[
        "selected_count_difference"
    ]


    if count_difference > 0:

        return "greedy_more_items"


    if count_difference < 0:

        return "greedy_fewer_items"


    return "same_count_different_items"


coverage_df[
    "failure_mechanism"
] = coverage_df.apply(
    classify_failure_mechanism,
    axis=1,
)


MECHANISM_ORDER = [
    "identical_subset",
    "same_count_different_items",
    "greedy_more_items",
    "greedy_fewer_items",
]


coverage_df[
    "failure_mechanism"
] = pd.Categorical(
    coverage_df[
        "failure_mechanism"
    ],
    categories=MECHANISM_ORDER,
    ordered=True,
)


coverage_df[
    "large_gap"
] = (
    coverage_df[
        "optimality_gap_percent"
    ]
    >
    10.0
)


coverage_df[
    "absolute_disclosure_cost_gap"
] = (
    coverage_df[
        "greedy_disclosure_cost"
    ]
    -
    coverage_df[
        "exact_disclosure_cost"
    ]
)


coverage_df[
    "utility_surplus_difference"
] = (
    (
        coverage_df[
            "greedy_utility"
        ]
        -
        coverage_df[
            "tau"
        ]
    )
    -
    (
        coverage_df[
            "exact_utility"
        ]
        -
        coverage_df[
            "tau"
        ]
    )
)


coverage_df[
    "privacy_difference"
] = (
    coverage_df[
        "greedy_privacy"
    ]
    -
    coverage_df[
        "exact_privacy"
    ]
)


coverage_df[
    "normalized_cognitive_difference"
] = (
    coverage_df[
        "greedy_normalized_cognitive_cost"
    ]
    -
    coverage_df[
        "exact_normalized_cognitive_cost"
    ]
)


# --------------------------------------------------
# Summary helper
# --------------------------------------------------

def summarize_mechanism(
    data_frame,
    total_rows,
):
    """
    Descriptive summary for one mechanism.
    """

    gap_values = data_frame[
        "optimality_gap_percent"
    ].to_numpy(
        dtype=float
    )


    large_gap_rows = int(
        data_frame[
            "large_gap"
        ].sum()
    )


    return pd.Series(
        {
            "rows":
                len(
                    data_frame
                ),

            "share_of_all_rows_percent":
                100.0
                *
                len(
                    data_frame
                )
                /
                total_rows,

            "mean_optimality_gap_percent":
                gap_values.mean(),

            "median_optimality_gap_percent":
                np.median(
                    gap_values
                ),

            "p90_optimality_gap_percent":
                np.quantile(
                    gap_values,
                    0.90,
                ),

            "p95_optimality_gap_percent":
                np.quantile(
                    gap_values,
                    0.95,
                ),

            "maximum_optimality_gap_percent":
                gap_values.max(),

            "large_gap_rows":
                large_gap_rows,

            "within_mechanism_large_gap_percent":
                100.0
                *
                large_gap_rows
                /
                len(
                    data_frame
                ),

            "mean_absolute_disclosure_cost_gap":
                data_frame[
                    "absolute_disclosure_cost_gap"
                ].mean(),

            "mean_jaccard_similarity":
                data_frame[
                    "jaccard_similarity"
                ].mean(),

            "mean_selected_count_difference":
                data_frame[
                    "selected_count_difference"
                ].mean(),

            "mean_privacy_difference":
                data_frame[
                    "privacy_difference"
                ].mean(),

            "mean_normalized_cognitive_difference":
                data_frame[
                    "normalized_cognitive_difference"
                ].mean(),

            "mean_utility_surplus_difference":
                data_frame[
                    "utility_surplus_difference"
                ].mean(),
        }
    )


# --------------------------------------------------
# Overall mechanism summary
# --------------------------------------------------

overall_mechanism_rows = []


for mechanism in MECHANISM_ORDER:

    current_df = coverage_df[
        coverage_df[
            "failure_mechanism"
        ]
        ==
        mechanism
    ]


    if len(
        current_df
    ) == 0:

        continue


    current_summary = summarize_mechanism(
        data_frame=current_df,
        total_rows=len(
            coverage_df
        ),
    )


    current_summary[
        "failure_mechanism"
    ] = mechanism


    overall_mechanism_rows.append(
        current_summary
    )


coverage_mechanism_summary_df = pd.DataFrame(
    overall_mechanism_rows
)


coverage_mechanism_summary_df = (
    coverage_mechanism_summary_df[
        [
            "failure_mechanism",
            "rows",
            "share_of_all_rows_percent",
            "mean_optimality_gap_percent",
            "median_optimality_gap_percent",
            "p90_optimality_gap_percent",
            "p95_optimality_gap_percent",
            "maximum_optimality_gap_percent",
            "large_gap_rows",
            "within_mechanism_large_gap_percent",
            "mean_absolute_disclosure_cost_gap",
            "mean_jaccard_similarity",
            "mean_selected_count_difference",
            "mean_privacy_difference",
            "mean_normalized_cognitive_difference",
            "mean_utility_surplus_difference",
        ]
    ]
)


# --------------------------------------------------
# Mechanism summary by structural regime
# --------------------------------------------------

regime_mechanism_rows = []


for regime in coverage_df[
    "regime"
].drop_duplicates():

    regime_df = coverage_df[
        coverage_df[
            "regime"
        ]
        ==
        regime
    ]


    for mechanism in MECHANISM_ORDER:

        current_df = regime_df[
            regime_df[
                "failure_mechanism"
            ]
            ==
            mechanism
        ]


        if len(
            current_df
        ) == 0:

            continue


        current_summary = summarize_mechanism(
            data_frame=current_df,
            total_rows=len(
                regime_df
            ),
        )


        current_summary[
            "regime"
        ] = regime


        current_summary[
            "failure_mechanism"
        ] = mechanism


        regime_mechanism_rows.append(
            current_summary
        )


coverage_mechanism_by_regime_df = pd.DataFrame(
    regime_mechanism_rows
)


coverage_mechanism_by_regime_df = (
    coverage_mechanism_by_regime_df[
        [
            "regime",
            "failure_mechanism",
            "rows",
            "share_of_all_rows_percent",
            "mean_optimality_gap_percent",
            "median_optimality_gap_percent",
            "p90_optimality_gap_percent",
            "p95_optimality_gap_percent",
            "maximum_optimality_gap_percent",
            "large_gap_rows",
            "within_mechanism_large_gap_percent",
            "mean_absolute_disclosure_cost_gap",
            "mean_jaccard_similarity",
            "mean_selected_count_difference",
            "mean_privacy_difference",
            "mean_normalized_cognitive_difference",
            "mean_utility_surplus_difference",
        ]
    ]
)


# --------------------------------------------------
# Large-gap attribution
# --------------------------------------------------

large_gap_df = coverage_df[
    coverage_df[
        "large_gap"
    ]
].copy()


large_gap_attribution_rows = []


for scope_name, scope_df in [
    (
        "overall",
        large_gap_df,
    ),
    *[
        (
            regime,
            large_gap_df[
                large_gap_df[
                    "regime"
                ]
                ==
                regime
            ],
        )
        for regime in coverage_df[
            "regime"
        ].drop_duplicates()
    ],
]:

    total_large_gap_rows = len(
        scope_df
    )


    for mechanism in MECHANISM_ORDER:

        current_rows = int(
            (
                scope_df[
                    "failure_mechanism"
                ]
                ==
                mechanism
            ).sum()
        )


        large_gap_attribution_rows.append(
            {
                "scope":
                    scope_name,

                "failure_mechanism":
                    mechanism,

                "large_gap_rows":
                    current_rows,

                "share_of_large_gap_rows_percent":
                    (
                        100.0
                        *
                        current_rows
                        /
                        total_large_gap_rows
                        if total_large_gap_rows > 0
                        else np.nan
                    ),
            }
        )


coverage_large_gap_attribution_df = pd.DataFrame(
    large_gap_attribution_rows
)


# --------------------------------------------------
# Extreme-tail analysis
# --------------------------------------------------

gap_p95 = float(
    coverage_df[
        "optimality_gap_percent"
    ].quantile(
        0.95
    )
)


gap_p99 = float(
    coverage_df[
        "optimality_gap_percent"
    ].quantile(
        0.99
    )
)


tail_definitions = [
    (
        "all_rows",
        coverage_df,
        0.0,
    ),

    (
        "gap_above_10_percent",
        coverage_df[
            coverage_df[
                "optimality_gap_percent"
            ]
            >
            10.0
        ],
        10.0,
    ),

    (
        "top_5_percent",
        coverage_df[
            coverage_df[
                "optimality_gap_percent"
            ]
            >=
            gap_p95
        ],
        gap_p95,
    ),

    (
        "top_1_percent",
        coverage_df[
            coverage_df[
                "optimality_gap_percent"
            ]
            >=
            gap_p99
        ],
        gap_p99,
    ),
]


tail_summary_rows = []


for tail_name, tail_df, threshold_value in (
    tail_definitions
):

    tail_summary_rows.append(
        {
            "tail_definition":
                tail_name,

            "gap_threshold_percent":
                threshold_value,

            "rows":
                len(
                    tail_df
                ),

            "share_of_all_rows_percent":
                100.0
                *
                len(
                    tail_df
                )
                /
                len(
                    coverage_df
                ),

            "mean_optimality_gap_percent":
                tail_df[
                    "optimality_gap_percent"
                ].mean(),

            "median_optimality_gap_percent":
                tail_df[
                    "optimality_gap_percent"
                ].median(),

            "maximum_optimality_gap_percent":
                tail_df[
                    "optimality_gap_percent"
                ].max(),

            "mean_absolute_disclosure_cost_gap":
                tail_df[
                    "absolute_disclosure_cost_gap"
                ].mean(),

            "mean_exact_disclosure_cost":
                tail_df[
                    "exact_disclosure_cost"
                ].mean(),

            "median_exact_disclosure_cost":
                tail_df[
                    "exact_disclosure_cost"
                ].median(),

            "same_count_different_items_percent":
                100.0
                *
                (
                    tail_df[
                        "failure_mechanism"
                    ]
                    ==
                    "same_count_different_items"
                ).mean(),

            "greedy_more_items_percent":
                100.0
                *
                (
                    tail_df[
                        "failure_mechanism"
                    ]
                    ==
                    "greedy_more_items"
                ).mean(),

            "greedy_fewer_items_percent":
                100.0
                *
                (
                    tail_df[
                        "failure_mechanism"
                    ]
                    ==
                    "greedy_fewer_items"
                ).mean(),

            "conflict_regime_percent":
                100.0
                *
                (
                    tail_df[
                        "regime"
                    ]
                    ==
                    "utility_burden_conflict"
                ).mean(),

            "mean_selected_count_difference":
                tail_df[
                    "selected_count_difference"
                ].mean(),

            "mean_jaccard_similarity":
                tail_df[
                    "jaccard_similarity"
                ].mean(),
        }
    )


coverage_tail_summary_df = pd.DataFrame(
    tail_summary_rows
)


# --------------------------------------------------
# Highest-gap individual conditions
# --------------------------------------------------

extreme_columns = [
    "matched_group_id",
    "n_items",
    "regime",
    "tau",
    "weight_profile",
    "failure_mechanism",
    "optimality_gap_percent",
    "absolute_disclosure_cost_gap",
    "exact_disclosure_cost",
    "greedy_disclosure_cost",
    "exact_selected_count",
    "greedy_selected_count",
    "selected_count_difference",
    "jaccard_similarity",
    "exact_utility",
    "greedy_utility",
    "exact_privacy",
    "greedy_privacy",
    "exact_normalized_cognitive_cost",
    "greedy_normalized_cognitive_cost",
    "exact_selected_items",
    "greedy_selected_items",
]


coverage_extreme_rows_df = (
    coverage_df
    .sort_values(
        "optimality_gap_percent",
        ascending=False,
    )[
        extreme_columns
    ]
    .head(
        20
    )
    .reset_index(
        drop=True
    )
)


# --------------------------------------------------
# Audit
# --------------------------------------------------

mechanism_counts = (
    coverage_df[
        "failure_mechanism"
    ]
    .value_counts(
        dropna=False
    )
)


large_gap_share_sum = (
    coverage_large_gap_attribution_df[
        coverage_large_gap_attribution_df[
            "scope"
        ]
        ==
        "overall"
    ][
        "share_of_large_gap_rows_percent"
    ]
    .sum()
)


coverage_mechanism_audit_df = pd.DataFrame(
    [
        {
            "benchmark_rows":
                len(
                    coverage_df
                ),

            "classified_rows":
                int(
                    mechanism_counts.sum()
                ),

            "unclassified_rows":
                int(
                    coverage_df[
                        "failure_mechanism"
                    ].isna().sum()
                ),

            "mechanism_categories_observed":
                int(
                    (
                        mechanism_counts
                        >
                        0
                    ).sum()
                ),

            "large_gap_rows":
                len(
                    large_gap_df
                ),

            "large_gap_percent":
                100.0
                *
                len(
                    large_gap_df
                )
                /
                len(
                    coverage_df
                ),

            "large_gap_attribution_share_sum":
                large_gap_share_sum,

            "p95_gap_threshold":
                gap_p95,

            "p99_gap_threshold":
                gap_p99,

            "maximum_gap_percent":
                coverage_df[
                    "optimality_gap_percent"
                ].max(),

            "minimum_absolute_gap":
                coverage_df[
                    "absolute_disclosure_cost_gap"
                ].min(),

            "negative_gap_rows":
                int(
                    (
                        coverage_df[
                            "optimality_gap_percent"
                        ]
                        <
                        -1e-8
                    ).sum()
                ),

            "count_difference_mismatch_rows":
                int(
                    (
                        (
                            coverage_df[
                                "greedy_selected_count"
                            ]
                            -
                            coverage_df[
                                "exact_selected_count"
                            ]
                        )
                        !=
                        coverage_df[
                            "selected_count_difference"
                        ]
                    ).sum()
                ),
        }
    ]
)


# --------------------------------------------------
# Integrity assertions
# --------------------------------------------------

assert coverage_mechanism_audit_df.loc[
    0,
    "classified_rows",
] == 24000


assert coverage_mechanism_audit_df.loc[
    0,
    "unclassified_rows",
] == 0


assert coverage_mechanism_audit_df.loc[
    0,
    "negative_gap_rows",
] == 0


assert coverage_mechanism_audit_df.loc[
    0,
    "count_difference_mismatch_rows",
] == 0


assert np.isclose(
    large_gap_share_sum,
    100.0,
    atol=1e-10,
)


assert (
    coverage_df.loc[
        coverage_df[
            "failure_mechanism"
        ]
        ==
        "identical_subset",
        "selected_count_difference",
    ]
    ==
    0
).all()


assert (
    coverage_df.loc[
        coverage_df[
            "failure_mechanism"
        ]
        ==
        "greedy_more_items",
        "selected_count_difference",
    ]
    >
    0
).all()


assert (
    coverage_df.loc[
        coverage_df[
            "failure_mechanism"
        ]
        ==
        "greedy_fewer_items",
        "selected_count_difference",
    ]
    <
    0
).all()


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

coverage_mechanism_summary_df.to_csv(
    COVERAGE_MECHANISM_SUMMARY_PATH,
    index=False,
)


coverage_mechanism_by_regime_df.to_csv(
    COVERAGE_MECHANISM_BY_REGIME_PATH,
    index=False,
)


coverage_large_gap_attribution_df.to_csv(
    COVERAGE_LARGE_GAP_ATTRIBUTION_PATH,
    index=False,
)


coverage_tail_summary_df.to_csv(
    COVERAGE_TAIL_SUMMARY_PATH,
    index=False,
)


coverage_extreme_rows_df.to_csv(
    COVERAGE_EXTREME_ROWS_PATH,
    index=False,
)


coverage_mechanism_audit_df.to_csv(
    COVERAGE_MECHANISM_AUDIT_PATH,
    index=False,
)


# --------------------------------------------------
# Printed report
# --------------------------------------------------

print(
    "======================================"
)

print(
    "COVERAGE FAILURE-MECHANISM AUDIT"
)

print(
    "======================================"
)


print(
    coverage_mechanism_audit_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.6f}",
    )
)


print(
    "\n======================================"
)

print(
    "COVERAGE FAILURE MECHANISMS — OVERALL"
)

print(
    "======================================"
)


print(
    coverage_mechanism_summary_df[
        [
            "failure_mechanism",
            "rows",
            "share_of_all_rows_percent",
            "mean_optimality_gap_percent",
            "median_optimality_gap_percent",
            "p95_optimality_gap_percent",
            "maximum_optimality_gap_percent",
            "large_gap_rows",
            "within_mechanism_large_gap_percent",
            "mean_absolute_disclosure_cost_gap",
            "mean_jaccard_similarity",
            "mean_selected_count_difference",
        ]
    ].to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


print(
    "\n======================================"
)

print(
    "COVERAGE LARGE-GAP MECHANISM ATTRIBUTION"
)

print(
    "======================================"
)


print(
    coverage_large_gap_attribution_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


print(
    "\n======================================"
)

print(
    "COVERAGE OPTIMALITY-GAP TAIL SUMMARY"
)

print(
    "======================================"
)


print(
    coverage_tail_summary_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


print(
    "\n======================================"
)

print(
    "TOP 20 COVERAGE OPTIMALITY-GAP CONDITIONS"
)

print(
    "======================================"
)


print(
    coverage_extreme_rows_df[
        [
            "matched_group_id",
            "n_items",
            "regime",
            "tau",
            "weight_profile",
            "failure_mechanism",
            "optimality_gap_percent",
            "absolute_disclosure_cost_gap",
            "exact_disclosure_cost",
            "greedy_disclosure_cost",
            "exact_selected_count",
            "greedy_selected_count",
            "jaccard_similarity",
        ]
    ].to_string(
        index=False,
        float_format=lambda value:
            f"{value:.6f}",
    )
)


print(
    "\nOverall mechanism summary saved to:"
)

print(
    COVERAGE_MECHANISM_SUMMARY_PATH
)


print(
    "\nMechanism-by-regime summary saved to:"
)

print(
    COVERAGE_MECHANISM_BY_REGIME_PATH
)


print(
    "\nLarge-gap attribution saved to:"
)

print(
    COVERAGE_LARGE_GAP_ATTRIBUTION_PATH
)


print(
    "\nTail summary saved to:"
)

print(
    COVERAGE_TAIL_SUMMARY_PATH
)


print(
    "\nExtreme rows saved to:"
)

print(
    COVERAGE_EXTREME_ROWS_PATH
)

# ===== notebook 02, code cell 90 =====
print('### notebook cell 90', flush=True)
import os
PROJECT_ROOT = os.environ.get("MST_PROJECT_ROOT", "..")
# --------------------------------------------------
# Step 132 — Coverage threshold effects, relative-gap
# denominator audit, and paired adjacent contrasts
# --------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

THRESHOLD_AUDIT_BOOTSTRAP_REPLICATES = 5000

THRESHOLD_AUDIT_BOOTSTRAP_SEED = (
    globals().get(
        "GLOBAL_SEED",
        2026,
    )
    + 132
)


THRESHOLD_ORDER = [
    0.70,
    0.80,
    0.90,
    0.95,
]


CANDIDATE_SIZE_ORDER = [
    8,
    10,
    12,
    14,
    16,
]


ADJACENT_THRESHOLD_CONTRASTS = [
    (
        0.80,
        0.70,
    ),

    (
        0.90,
        0.80,
    ),

    (
        0.95,
        0.90,
    ),
]


EXACT_RECOVERY_TOLERANCE_PERCENT = 1e-10


METRIC_NAMES = [
    "mean_optimality_gap_percent",
    "mean_absolute_disclosure_cost_gap",
    "mean_exact_disclosure_cost",
    "exact_cost_recovery_percent",
    "large_gap_percent",
    "more_items_percent",
    "same_count_different_items_percent",
    "mean_jaccard_similarity",
    "mean_selected_count_difference",
]


AGGREGATE_COLUMNS = [
    "comparable_rows",
    "gap_sum",
    "absolute_gap_sum",
    "exact_cost_sum",
    "exact_recovery_sum",
    "large_gap_sum",
    "more_items_sum",
    "same_count_different_items_sum",
    "jaccard_sum",
    "selected_count_difference_sum",
]


# --------------------------------------------------
# Paths
# --------------------------------------------------

if "RESULTS_DIR" not in globals():

    RESULTS_DIR = Path(
        os.path.join(PROJECT_ROOT, "05_Results")
    )


if "TABLES_DIR" not in globals():

    TABLES_DIR = Path(
        os.path.join(PROJECT_ROOT, "07_Tables")
    )


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


COVERAGE_MAIN_RESULTS_PATH = (
    RESULTS_DIR
    / (
        "coverage_exact_vs_greedy_"
        "main_benchmark_v1_pool_normalized.csv"
    )
)


THRESHOLD_DESCRIPTIVE_PATH = (
    TABLES_DIR
    / "coverage_threshold_denominator_descriptive_summary.csv"
)


THRESHOLD_TAIL_COMPOSITION_PATH = (
    TABLES_DIR
    / "coverage_threshold_extreme_tail_composition.csv"
)


THRESHOLD_CONTRAST_PATH = (
    TABLES_DIR
    / "coverage_threshold_paired_adjacent_contrasts.csv"
)


THRESHOLD_BOOTSTRAP_REPLICATES_PATH = (
    RESULTS_DIR
    / "coverage_threshold_paired_bootstrap_replicates.csv"
)


THRESHOLD_AUDIT_PATH = (
    RESULTS_DIR
    / "coverage_threshold_denominator_audit.csv"
)


# --------------------------------------------------
# Load benchmark
# --------------------------------------------------

coverage_df = pd.read_csv(
    COVERAGE_MAIN_RESULTS_PATH,
    low_memory=False,
)


required_columns = [
    "matched_group_id",
    "n_items",
    "regime",
    "tau",
    "weight_profile",
    "exact_status",
    "greedy_status",
    "exact_selected_items",
    "greedy_selected_items",
    "exact_selected_count",
    "greedy_selected_count",
    "exact_disclosure_cost",
    "greedy_disclosure_cost",
    "optimality_gap_percent",
    "jaccard_similarity",
    "selected_count_difference",
]


missing_columns = [
    column_name
    for column_name in required_columns
    if column_name not in coverage_df.columns
]


if missing_columns:

    raise KeyError(
        "Missing required columns: "
        + ", ".join(
            missing_columns
        )
    )


numeric_columns = [
    "n_items",
    "tau",
    "exact_selected_count",
    "greedy_selected_count",
    "exact_disclosure_cost",
    "greedy_disclosure_cost",
    "optimality_gap_percent",
    "jaccard_similarity",
    "selected_count_difference",
]


for column_name in numeric_columns:

    coverage_df[
        column_name
    ] = pd.to_numeric(
        coverage_df[
            column_name
        ],
        errors="coerce",
    )


# --------------------------------------------------
# Benchmark integrity
# --------------------------------------------------

assert len(
    coverage_df
) == 24000


assert (
    coverage_df[
        "matched_group_id"
    ].nunique()
    == 500
)


assert (
    sorted(
        coverage_df[
            "tau"
        ]
        .dropna()
        .unique()
        .tolist()
    )
    ==
    THRESHOLD_ORDER
)


assert (
    sorted(
        coverage_df[
            "n_items"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    ==
    CANDIDATE_SIZE_ORDER
)


assert (
    coverage_df[
        "exact_status"
    ]
    ==
    "OPTIMAL"
).all()


assert (
    coverage_df[
        "greedy_status"
    ]
    ==
    "FEASIBLE"
).all()


# --------------------------------------------------
# Derived indicators
# --------------------------------------------------

coverage_df[
    "exact_selected_items_clean"
] = (
    coverage_df[
        "exact_selected_items"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
)


coverage_df[
    "greedy_selected_items_clean"
] = (
    coverage_df[
        "greedy_selected_items"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
)


coverage_df[
    "same_selected_subset"
] = (
    coverage_df[
        "exact_selected_items_clean"
    ]
    ==
    coverage_df[
        "greedy_selected_items_clean"
    ]
)


coverage_df[
    "exact_cost_recovery_numeric"
] = (
    coverage_df[
        "optimality_gap_percent"
    ]
    .abs()
    <=
    EXACT_RECOVERY_TOLERANCE_PERCENT
).astype(int)


coverage_df[
    "large_gap_numeric"
] = (
    coverage_df[
        "optimality_gap_percent"
    ]
    >
    10.0
).astype(int)


coverage_df[
    "more_items_numeric"
] = (
    coverage_df[
        "selected_count_difference"
    ]
    >
    0
).astype(int)


coverage_df[
    "same_count_different_items_numeric"
] = (
    (
        coverage_df[
            "selected_count_difference"
        ]
        ==
        0
    )
    &
    (
        ~coverage_df[
            "same_selected_subset"
        ]
    )
).astype(int)


coverage_df[
    "absolute_disclosure_cost_gap"
] = (
    coverage_df[
        "greedy_disclosure_cost"
    ]
    -
    coverage_df[
        "exact_disclosure_cost"
    ]
)


# Remove negligible floating-point deviations.
coverage_df.loc[
    coverage_df[
        "absolute_disclosure_cost_gap"
    ].abs()
    <=
    1e-12,
    "absolute_disclosure_cost_gap",
] = 0.0


# --------------------------------------------------
# Global tail thresholds
# --------------------------------------------------

GLOBAL_P95_GAP = float(
    coverage_df[
        "optimality_gap_percent"
    ].quantile(
        0.95
    )
)


GLOBAL_P99_GAP = float(
    coverage_df[
        "optimality_gap_percent"
    ].quantile(
        0.99
    )
)


coverage_df[
    "global_top_5_percent"
] = (
    coverage_df[
        "optimality_gap_percent"
    ]
    >=
    GLOBAL_P95_GAP
)


coverage_df[
    "global_top_1_percent"
] = (
    coverage_df[
        "optimality_gap_percent"
    ]
    >=
    GLOBAL_P99_GAP
)


# --------------------------------------------------
# Descriptive threshold summary
# --------------------------------------------------

descriptive_rows = []


for tau in THRESHOLD_ORDER:

    current_df = coverage_df[
        np.isclose(
            coverage_df[
                "tau"
            ],
            tau,
        )
    ]


    gap_values = (
        current_df[
            "optimality_gap_percent"
        ]
        .to_numpy(
            dtype=float
        )
    )


    descriptive_rows.append(
        {
            "tau":
                tau,

            "rows":
                len(
                    current_df
                ),

            "matched_groups":
                current_df[
                    "matched_group_id"
                ].nunique(),

            "mean_optimality_gap_percent":
                gap_values.mean(),

            "median_optimality_gap_percent":
                np.median(
                    gap_values
                ),

            "p90_optimality_gap_percent":
                np.quantile(
                    gap_values,
                    0.90,
                ),

            "p95_optimality_gap_percent":
                np.quantile(
                    gap_values,
                    0.95,
                ),

            "maximum_optimality_gap_percent":
                gap_values.max(),

            "mean_absolute_disclosure_cost_gap":
                current_df[
                    "absolute_disclosure_cost_gap"
                ].mean(),

            "median_absolute_disclosure_cost_gap":
                current_df[
                    "absolute_disclosure_cost_gap"
                ].median(),

            "mean_exact_disclosure_cost":
                current_df[
                    "exact_disclosure_cost"
                ].mean(),

            "median_exact_disclosure_cost":
                current_df[
                    "exact_disclosure_cost"
                ].median(),

            "exact_cost_recovery_percent":
                100.0
                *
                current_df[
                    "exact_cost_recovery_numeric"
                ].mean(),

            "large_gap_percent":
                100.0
                *
                current_df[
                    "large_gap_numeric"
                ].mean(),

            "more_items_percent":
                100.0
                *
                current_df[
                    "more_items_numeric"
                ].mean(),

            "same_count_different_items_percent":
                100.0
                *
                current_df[
                    "same_count_different_items_numeric"
                ].mean(),

            "mean_jaccard_similarity":
                current_df[
                    "jaccard_similarity"
                ].mean(),

            "mean_selected_count_difference":
                current_df[
                    "selected_count_difference"
                ].mean(),

            "global_top_5_percent_rows":
                int(
                    current_df[
                        "global_top_5_percent"
                    ].sum()
                ),

            "global_top_1_percent_rows":
                int(
                    current_df[
                        "global_top_1_percent"
                    ].sum()
                ),

            "within_threshold_top_5_percent_rate":
                100.0
                *
                current_df[
                    "global_top_5_percent"
                ].mean(),

            "within_threshold_top_1_percent_rate":
                100.0
                *
                current_df[
                    "global_top_1_percent"
                ].mean(),
        }
    )


threshold_descriptive_df = pd.DataFrame(
    descriptive_rows
)


# --------------------------------------------------
# Tail composition by threshold
# --------------------------------------------------

tail_definitions = [
    (
        "gap_above_10_percent",
        coverage_df[
            coverage_df[
                "optimality_gap_percent"
            ]
            >
            10.0
        ],
        10.0,
    ),

    (
        "global_top_5_percent",
        coverage_df[
            coverage_df[
                "global_top_5_percent"
            ]
        ],
        GLOBAL_P95_GAP,
    ),

    (
        "global_top_1_percent",
        coverage_df[
            coverage_df[
                "global_top_1_percent"
            ]
        ],
        GLOBAL_P99_GAP,
    ),
]


tail_composition_rows = []


for tail_name, tail_df, gap_threshold in (
    tail_definitions
):

    total_tail_rows = len(
        tail_df
    )


    for tau in THRESHOLD_ORDER:

        current_df = tail_df[
            np.isclose(
                tail_df[
                    "tau"
                ],
                tau,
            )
        ]


        tau_total_rows = int(
            np.isclose(
                coverage_df[
                    "tau"
                ],
                tau,
            ).sum()
        )


        tail_composition_rows.append(
            {
                "tail_definition":
                    tail_name,

                "gap_threshold_percent":
                    gap_threshold,

                "tau":
                    tau,

                "tail_rows":
                    len(
                        current_df
                    ),

                "share_of_tail_percent":
                    (
                        100.0
                        *
                        len(
                            current_df
                        )
                        /
                        total_tail_rows
                    ),

                "share_of_threshold_rows_percent":
                    (
                        100.0
                        *
                        len(
                            current_df
                        )
                        /
                        tau_total_rows
                    ),

                "mean_optimality_gap_percent":
                    current_df[
                        "optimality_gap_percent"
                    ].mean(),

                "mean_absolute_disclosure_cost_gap":
                    current_df[
                        "absolute_disclosure_cost_gap"
                    ].mean(),

                "mean_exact_disclosure_cost":
                    current_df[
                        "exact_disclosure_cost"
                    ].mean(),

                "more_items_percent":
                    100.0
                    *
                    current_df[
                        "more_items_numeric"
                    ].mean(),

                "same_count_different_items_percent":
                    100.0
                    *
                    current_df[
                        "same_count_different_items_numeric"
                    ].mean(),

                "mean_selected_count_difference":
                    current_df[
                        "selected_count_difference"
                    ].mean(),
            }
        )


threshold_tail_composition_df = pd.DataFrame(
    tail_composition_rows
)


# --------------------------------------------------
# Matched-group sufficient statistics
#
# Each group × threshold contains:
# 3 regimes × 4 weight profiles = 12 conditions.
# --------------------------------------------------

cluster_df = (
    coverage_df
    .groupby(
        [
            "n_items",
            "matched_group_id",
            "tau",
        ],
        as_index=False,
    )
    .agg(
        comparable_rows=(
            "optimality_gap_percent",
            "size",
        ),

        gap_sum=(
            "optimality_gap_percent",
            "sum",
        ),

        absolute_gap_sum=(
            "absolute_disclosure_cost_gap",
            "sum",
        ),

        exact_cost_sum=(
            "exact_disclosure_cost",
            "sum",
        ),

        exact_recovery_sum=(
            "exact_cost_recovery_numeric",
            "sum",
        ),

        large_gap_sum=(
            "large_gap_numeric",
            "sum",
        ),

        more_items_sum=(
            "more_items_numeric",
            "sum",
        ),

        same_count_different_items_sum=(
            "same_count_different_items_numeric",
            "sum",
        ),

        jaccard_sum=(
            "jaccard_similarity",
            "sum",
        ),

        selected_count_difference_sum=(
            "selected_count_difference",
            "sum",
        ),
    )
)


assert (
    cluster_df[
        "comparable_rows"
    ]
    ==
    12
).all()


# --------------------------------------------------
# Metric conversion
# --------------------------------------------------

def calculate_threshold_metrics(
    totals,
):
    """
    Convert sufficient-statistic totals into
    threshold-level metrics.
    """

    comparable_rows = totals[0]


    if comparable_rows <= 0:

        return np.full(
            len(
                METRIC_NAMES
            ),
            np.nan,
        )


    return np.array(
        [
            totals[1]
            /
            comparable_rows,

            totals[2]
            /
            comparable_rows,

            totals[3]
            /
            comparable_rows,

            100.0
            *
            totals[4]
            /
            comparable_rows,

            100.0
            *
            totals[5]
            /
            comparable_rows,

            100.0
            *
            totals[6]
            /
            comparable_rows,

            100.0
            *
            totals[7]
            /
            comparable_rows,

            totals[8]
            /
            comparable_rows,

            totals[9]
            /
            comparable_rows,
        ],
        dtype=float,
    )


# --------------------------------------------------
# Aligned arrays by size and threshold
# --------------------------------------------------

arrays_by_size_threshold = {}


for n_items in CANDIDATE_SIZE_ORDER:

    arrays_by_size_threshold[
        n_items
    ] = {}


    reference_group_ids = None


    for tau in THRESHOLD_ORDER:

        current_df = (
            cluster_df[
                (
                    cluster_df[
                        "n_items"
                    ]
                    ==
                    n_items
                )
                &
                (
                    np.isclose(
                        cluster_df[
                            "tau"
                        ],
                        tau,
                    )
                )
            ]
            .sort_values(
                "matched_group_id"
            )
            .reset_index(
                drop=True
            )
        )


        current_group_ids = (
            current_df[
                "matched_group_id"
            ].tolist()
        )


        assert len(
            current_group_ids
        ) == 100


        if reference_group_ids is None:

            reference_group_ids = (
                current_group_ids
            )

        else:

            assert (
                current_group_ids
                ==
                reference_group_ids
            )


        arrays_by_size_threshold[
            n_items
        ][
            tau
        ] = (
            current_df[
                AGGREGATE_COLUMNS
            ]
            .to_numpy(
                dtype=float
            )
        )


# --------------------------------------------------
# Observed threshold totals
# --------------------------------------------------

observed_totals = {
    tau: np.zeros(
        len(
            AGGREGATE_COLUMNS
        ),
        dtype=float,
    )
    for tau in THRESHOLD_ORDER
}


for n_items in CANDIDATE_SIZE_ORDER:

    for tau in THRESHOLD_ORDER:

        observed_totals[
            tau
        ] += (
            arrays_by_size_threshold[
                n_items
            ][
                tau
            ]
            .sum(
                axis=0
            )
        )


observed_metrics = {
    tau:
        calculate_threshold_metrics(
            observed_totals[
                tau
            ]
        )
    for tau in THRESHOLD_ORDER
}


# --------------------------------------------------
# Shared paired bootstrap across all thresholds
# --------------------------------------------------

bootstrap_rng = np.random.default_rng(
    THRESHOLD_AUDIT_BOOTSTRAP_SEED
)


bootstrap_differences = np.empty(
    (
        THRESHOLD_AUDIT_BOOTSTRAP_REPLICATES,
        len(
            ADJACENT_THRESHOLD_CONTRASTS
        ),
        len(
            METRIC_NAMES
        ),
    ),
    dtype=float,
)


for bootstrap_index in range(
    THRESHOLD_AUDIT_BOOTSTRAP_REPLICATES
):

    bootstrap_totals = {
        tau: np.zeros(
            len(
                AGGREGATE_COLUMNS
            ),
            dtype=float,
        )
        for tau in THRESHOLD_ORDER
    }


    for n_items in CANDIDATE_SIZE_ORDER:

        number_of_groups = (
            arrays_by_size_threshold[
                n_items
            ][
                THRESHOLD_ORDER[0]
            ].shape[0]
        )


        sampled_indices = (
            bootstrap_rng.integers(
                low=0,
                high=number_of_groups,
                size=number_of_groups,
            )
        )


        for tau in THRESHOLD_ORDER:

            bootstrap_totals[
                tau
            ] += (
                arrays_by_size_threshold[
                    n_items
                ][
                    tau
                ][
                    sampled_indices
                ]
                .sum(
                    axis=0
                )
            )


    bootstrap_metrics = {
        tau:
            calculate_threshold_metrics(
                bootstrap_totals[
                    tau
                ]
            )
        for tau in THRESHOLD_ORDER
    }


    for contrast_index, (
        higher_tau,
        lower_tau,
    ) in enumerate(
        ADJACENT_THRESHOLD_CONTRASTS
    ):

        bootstrap_differences[
            bootstrap_index,
            contrast_index,
        ] = (
            bootstrap_metrics[
                higher_tau
            ]
            -
            bootstrap_metrics[
                lower_tau
            ]
        )


# --------------------------------------------------
# Paired adjacent-threshold contrasts
# --------------------------------------------------

contrast_rows = []


for contrast_index, (
    higher_tau,
    lower_tau,
) in enumerate(
    ADJACENT_THRESHOLD_CONTRASTS
):

    observed_difference = (
        observed_metrics[
            higher_tau
        ]
        -
        observed_metrics[
            lower_tau
        ]
    )


    current_bootstrap = (
        bootstrap_differences[
            :,
            contrast_index,
            :,
        ]
    )


    confidence_lower = np.quantile(
        current_bootstrap,
        0.025,
        axis=0,
    )


    confidence_upper = np.quantile(
        current_bootstrap,
        0.975,
        axis=0,
    )


    standard_errors = (
        current_bootstrap.std(
            axis=0,
            ddof=1,
        )
    )


    for metric_index, metric_name in enumerate(
        METRIC_NAMES
    ):

        contrast_rows.append(
            {
                "higher_tau":
                    higher_tau,

                "lower_tau":
                    lower_tau,

                "contrast":
                    (
                        f"{higher_tau:.2f}"
                        f" minus "
                        f"{lower_tau:.2f}"
                    ),

                "metric":
                    metric_name,

                "estimated_difference":
                    observed_difference[
                        metric_index
                    ],

                "bootstrap_standard_error":
                    standard_errors[
                        metric_index
                    ],

                "ci_95_lower":
                    confidence_lower[
                        metric_index
                    ],

                "ci_95_upper":
                    confidence_upper[
                        metric_index
                    ],

                "ci_excludes_zero":
                    (
                        confidence_lower[
                            metric_index
                        ]
                        >
                        0.0
                    )
                    or
                    (
                        confidence_upper[
                            metric_index
                        ]
                        <
                        0.0
                    ),

                "matched_groups":
                    500,

                "matched_groups_per_size":
                    100,

                "bootstrap_replicates":
                    THRESHOLD_AUDIT_BOOTSTRAP_REPLICATES,

                "resampling_design":
                    (
                        "paired_across_thresholds_"
                        "stratified_by_candidate_set_size"
                    ),

                "multiple_comparison_adjustment":
                    "none",
            }
        )


threshold_contrast_df = pd.DataFrame(
    contrast_rows
)


# --------------------------------------------------
# Save bootstrap replicates
# --------------------------------------------------

replicate_rows = []


for bootstrap_index in range(
    THRESHOLD_AUDIT_BOOTSTRAP_REPLICATES
):

    for contrast_index, (
        higher_tau,
        lower_tau,
    ) in enumerate(
        ADJACENT_THRESHOLD_CONTRASTS
    ):

        replicate_row = {
            "bootstrap_replicate":
                bootstrap_index,

            "higher_tau":
                higher_tau,

            "lower_tau":
                lower_tau,

            "contrast":
                (
                    f"{higher_tau:.2f}"
                    f" minus "
                    f"{lower_tau:.2f}"
                ),
        }


        for metric_index, metric_name in enumerate(
            METRIC_NAMES
        ):

            replicate_row[
                metric_name
            ] = (
                bootstrap_differences[
                    bootstrap_index,
                    contrast_index,
                    metric_index,
                ]
            )


        replicate_rows.append(
            replicate_row
        )


threshold_bootstrap_replicates_df = (
    pd.DataFrame(
        replicate_rows
    )
)


# --------------------------------------------------
# Audit
# --------------------------------------------------

group_threshold_counts = (
    coverage_df
    .groupby(
        [
            "matched_group_id",
            "tau",
        ]
    )
    .size()
)


threshold_audit_df = pd.DataFrame(
    [
        {
            "benchmark_rows":
                len(
                    coverage_df
                ),

            "matched_groups":
                coverage_df[
                    "matched_group_id"
                ].nunique(),

            "thresholds":
                coverage_df[
                    "tau"
                ].nunique(),

            "minimum_rows_per_group_threshold":
                int(
                    group_threshold_counts.min()
                ),

            "maximum_rows_per_group_threshold":
                int(
                    group_threshold_counts.max()
                ),

            "global_p95_gap_threshold":
                GLOBAL_P95_GAP,

            "global_p99_gap_threshold":
                GLOBAL_P99_GAP,

            "negative_absolute_gap_rows":
                int(
                    (
                        coverage_df[
                            "absolute_disclosure_cost_gap"
                        ]
                        <
                        -1e-10
                    ).sum()
                ),

            "maximum_reconstruction_error":
                (
                    (
                        coverage_df[
                            "absolute_disclosure_cost_gap"
                        ]
                        /
                        coverage_df[
                            "exact_disclosure_cost"
                        ]
                    )
                    *
                    100.0
                    -
                    coverage_df[
                        "optimality_gap_percent"
                    ]
                )
                .abs()
                .max(),

            "bootstrap_replicates":
                THRESHOLD_AUDIT_BOOTSTRAP_REPLICATES,

            "confidence_intervals":
                "unadjusted_percentile_95",
        }
    ]
)


assert (
    group_threshold_counts
    ==
    12
).all()


assert threshold_audit_df.loc[
    0,
    "negative_absolute_gap_rows",
] == 0


assert threshold_audit_df.loc[
    0,
    "maximum_reconstruction_error",
] <= 1e-8


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

threshold_descriptive_df.to_csv(
    THRESHOLD_DESCRIPTIVE_PATH,
    index=False,
)


threshold_tail_composition_df.to_csv(
    THRESHOLD_TAIL_COMPOSITION_PATH,
    index=False,
)


threshold_contrast_df.to_csv(
    THRESHOLD_CONTRAST_PATH,
    index=False,
)


threshold_bootstrap_replicates_df.to_csv(
    THRESHOLD_BOOTSTRAP_REPLICATES_PATH,
    index=False,
)


threshold_audit_df.to_csv(
    THRESHOLD_AUDIT_PATH,
    index=False,
)


# --------------------------------------------------
# Readable contrast display
# --------------------------------------------------

contrast_display_df = (
    threshold_contrast_df.copy()
)


contrast_display_df[
    "difference_and_ci"
] = (
    contrast_display_df[
        "estimated_difference"
    ].map(
        lambda value:
            f"{value:.4f}"
    )
    +
    " ["
    +
    contrast_display_df[
        "ci_95_lower"
    ].map(
        lambda value:
            f"{value:.4f}"
    )
    +
    ", "
    +
    contrast_display_df[
        "ci_95_upper"
    ].map(
        lambda value:
            f"{value:.4f}"
    )
    +
    "]"
)


# --------------------------------------------------
# Printed report
# --------------------------------------------------

print(
    "======================================"
)

print(
    "COVERAGE THRESHOLD-DENOMINATOR AUDIT"
)

print(
    "======================================"
)


print(
    threshold_audit_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.10f}",
    )
)


print(
    "\n======================================"
)

print(
    "COVERAGE RESULTS BY SUFFICIENCY THRESHOLD"
)

print(
    "======================================"
)


print(
    threshold_descriptive_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


print(
    "\n======================================"
)

print(
    "EXTREME-TAIL COMPOSITION BY THRESHOLD"
)

print(
    "======================================"
)


print(
    threshold_tail_composition_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


print(
    "\n======================================"
)

print(
    "PAIRED ADJACENT-THRESHOLD CONTRASTS"
)

print(
    "======================================"
)


print(
    contrast_display_df[
        [
            "contrast",
            "metric",
            "difference_and_ci",
            "ci_excludes_zero",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nThreshold descriptive summary saved to:"
)

print(
    THRESHOLD_DESCRIPTIVE_PATH
)


print(
    "\nTail composition saved to:"
)

print(
    THRESHOLD_TAIL_COMPOSITION_PATH
)


print(
    "\nPaired threshold contrasts saved to:"
)

print(
    THRESHOLD_CONTRAST_PATH
)


print(
    "\nBootstrap replicates saved to:"
)

print(
    THRESHOLD_BOOTSTRAP_REPLICATES_PATH
)


print(
    "\nAudit saved to:"
)

print(
    THRESHOLD_AUDIT_PATH
)

# ===== notebook 02, code cell 91 =====
print('### notebook cell 91', flush=True)
import os
PROJECT_ROOT = os.environ.get("MST_PROJECT_ROOT", "..")
# --------------------------------------------------
# Step 133 — Coverage benchmark sensitivity by
# candidate-set size and objective-weight profile
# --------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SENSITIVITY_BOOTSTRAP_REPLICATES = 5000

SENSITIVITY_BOOTSTRAP_SEED = (
    globals().get(
        "GLOBAL_SEED",
        2026,
    )
    + 133
)


CANDIDATE_SIZE_ORDER = [
    8,
    10,
    12,
    14,
    16,
]


WEIGHT_PROFILE_ORDER = [
    "balanced",
    "privacy_priority",
    "interpretation_priority",
    "parsimony_priority",
]


REFERENCE_WEIGHT_PROFILE = "balanced"


WEIGHT_PROFILE_CONTRASTS = [
    (
        "privacy_priority",
        REFERENCE_WEIGHT_PROFILE,
    ),

    (
        "interpretation_priority",
        REFERENCE_WEIGHT_PROFILE,
    ),

    (
        "parsimony_priority",
        REFERENCE_WEIGHT_PROFILE,
    ),
]


METRIC_NAMES = [
    "mean_optimality_gap_percent",
    "exact_cost_recovery_percent",
    "mean_jaccard_similarity",
    "large_gap_percent",
    "more_items_percent",
    "mean_selected_count_difference",
]


AGGREGATE_COLUMNS = [
    "comparable_rows",
    "gap_sum",
    "exact_recovery_sum",
    "jaccard_sum",
    "large_gap_sum",
    "more_items_sum",
    "selected_count_difference_sum",
]


EXACT_RECOVERY_TOLERANCE_PERCENT = 1e-10


# --------------------------------------------------
# Paths
# --------------------------------------------------

if "RESULTS_DIR" not in globals():

    RESULTS_DIR = Path(
        os.path.join(PROJECT_ROOT, "05_Results")
    )


if "TABLES_DIR" not in globals():

    TABLES_DIR = Path(
        os.path.join(PROJECT_ROOT, "07_Tables")
    )


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


COVERAGE_MAIN_RESULTS_PATH = (
    RESULTS_DIR
    / (
        "coverage_exact_vs_greedy_"
        "main_benchmark_v1_pool_normalized.csv"
    )
)


SIZE_SUMMARY_PATH = (
    TABLES_DIR
    / "coverage_candidate_size_bootstrap_summary.csv"
)


SIZE_BOOTSTRAP_REPLICATES_PATH = (
    RESULTS_DIR
    / "coverage_candidate_size_bootstrap_replicates.csv"
)


WEIGHT_DESCRIPTIVE_PATH = (
    TABLES_DIR
    / "coverage_weight_profile_descriptive_summary.csv"
)


WEIGHT_CONTRAST_PATH = (
    TABLES_DIR
    / "coverage_paired_weight_profile_contrasts.csv"
)


WEIGHT_BOOTSTRAP_REPLICATES_PATH = (
    RESULTS_DIR
    / "coverage_weight_profile_bootstrap_replicates.csv"
)


SENSITIVITY_AUDIT_PATH = (
    RESULTS_DIR
    / "coverage_size_weight_sensitivity_audit.csv"
)


# --------------------------------------------------
# Load benchmark
# --------------------------------------------------

coverage_df = pd.read_csv(
    COVERAGE_MAIN_RESULTS_PATH,
    low_memory=False,
)


required_columns = [
    "matched_group_id",
    "n_items",
    "regime",
    "tau",
    "weight_profile",
    "exact_status",
    "greedy_status",
    "optimality_gap_percent",
    "jaccard_similarity",
    "selected_count_difference",
]


missing_columns = [
    column_name
    for column_name in required_columns
    if column_name not in coverage_df.columns
]


if missing_columns:

    raise KeyError(
        "Missing required columns: "
        + ", ".join(
            missing_columns
        )
    )


for column_name in [
    "n_items",
    "tau",
    "optimality_gap_percent",
    "jaccard_similarity",
    "selected_count_difference",
]:

    coverage_df[
        column_name
    ] = pd.to_numeric(
        coverage_df[
            column_name
        ],
        errors="coerce",
    )


# --------------------------------------------------
# Design validation
# --------------------------------------------------

assert len(
    coverage_df
) == 24000


assert (
    coverage_df[
        "matched_group_id"
    ].nunique()
    == 500
)


assert (
    sorted(
        coverage_df[
            "n_items"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    ==
    CANDIDATE_SIZE_ORDER
)


assert set(
    coverage_df[
        "weight_profile"
    ].dropna().unique()
) == set(
    WEIGHT_PROFILE_ORDER
)


assert (
    coverage_df[
        "exact_status"
    ]
    ==
    "OPTIMAL"
).all()


assert (
    coverage_df[
        "greedy_status"
    ]
    ==
    "FEASIBLE"
).all()


# --------------------------------------------------
# Reporting indicators
# --------------------------------------------------

coverage_df[
    "exact_recovery_numeric"
] = (
    coverage_df[
        "optimality_gap_percent"
    ]
    .abs()
    <=
    EXACT_RECOVERY_TOLERANCE_PERCENT
).astype(int)


coverage_df[
    "large_gap_numeric"
] = (
    coverage_df[
        "optimality_gap_percent"
    ]
    >
    10.0
).astype(int)


coverage_df[
    "more_items_numeric"
] = (
    coverage_df[
        "selected_count_difference"
    ]
    >
    0
).astype(int)


# --------------------------------------------------
# Metric conversion
# --------------------------------------------------

def calculate_metrics(
    totals,
):
    """
    Convert sufficient-statistic totals into the six
    Exact-relative Dynamic Greedy metrics.
    """

    comparable_rows = totals[0]


    if comparable_rows <= 0:

        return np.full(
            len(
                METRIC_NAMES
            ),
            np.nan,
        )


    return np.array(
        [
            totals[1]
            /
            comparable_rows,

            100.0
            *
            totals[2]
            /
            comparable_rows,

            totals[3]
            /
            comparable_rows,

            100.0
            *
            totals[4]
            /
            comparable_rows,

            100.0
            *
            totals[5]
            /
            comparable_rows,

            totals[6]
            /
            comparable_rows,
        ],
        dtype=float,
    )


# ==================================================
# PART A — Candidate-set size sensitivity
# ==================================================

size_cluster_df = (
    coverage_df
    .groupby(
        [
            "n_items",
            "matched_group_id",
        ],
        as_index=False,
    )
    .agg(
        comparable_rows=(
            "optimality_gap_percent",
            "size",
        ),

        gap_sum=(
            "optimality_gap_percent",
            "sum",
        ),

        exact_recovery_sum=(
            "exact_recovery_numeric",
            "sum",
        ),

        jaccard_sum=(
            "jaccard_similarity",
            "sum",
        ),

        large_gap_sum=(
            "large_gap_numeric",
            "sum",
        ),

        more_items_sum=(
            "more_items_numeric",
            "sum",
        ),

        selected_count_difference_sum=(
            "selected_count_difference",
            "sum",
        ),
    )
)


# Each matched group contains:
# 3 regimes × 4 thresholds × 4 profiles = 48 rows.
assert (
    size_cluster_df[
        "comparable_rows"
    ]
    ==
    48
).all()


size_arrays = {}


for n_items in CANDIDATE_SIZE_ORDER:

    current_df = (
        size_cluster_df[
            size_cluster_df[
                "n_items"
            ]
            ==
            n_items
        ]
        .sort_values(
            "matched_group_id"
        )
        .reset_index(
            drop=True
        )
    )


    assert len(
        current_df
    ) == 100


    size_arrays[
        n_items
    ] = (
        current_df[
            AGGREGATE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )


size_observed_metrics = {}


for n_items in CANDIDATE_SIZE_ORDER:

    size_observed_metrics[
        n_items
    ] = calculate_metrics(
        size_arrays[
            n_items
        ].sum(
            axis=0
        )
    )


size_bootstrap_rng = np.random.default_rng(
    SENSITIVITY_BOOTSTRAP_SEED
)


size_bootstrap_values = np.empty(
    (
        SENSITIVITY_BOOTSTRAP_REPLICATES,
        len(
            CANDIDATE_SIZE_ORDER
        ),
        len(
            METRIC_NAMES
        ),
    ),
    dtype=float,
)


for bootstrap_index in range(
    SENSITIVITY_BOOTSTRAP_REPLICATES
):

    for size_index, n_items in enumerate(
        CANDIDATE_SIZE_ORDER
    ):

        number_of_groups = (
            size_arrays[
                n_items
            ].shape[0]
        )


        sampled_indices = (
            size_bootstrap_rng.integers(
                low=0,
                high=number_of_groups,
                size=number_of_groups,
            )
        )


        resampled_totals = (
            size_arrays[
                n_items
            ][
                sampled_indices
            ]
            .sum(
                axis=0
            )
        )


        size_bootstrap_values[
            bootstrap_index,
            size_index,
        ] = calculate_metrics(
            resampled_totals
        )


size_summary_rows = []


for size_index, n_items in enumerate(
    CANDIDATE_SIZE_ORDER
):

    current_bootstrap = (
        size_bootstrap_values[
            :,
            size_index,
            :,
        ]
    )


    confidence_lower = np.quantile(
        current_bootstrap,
        0.025,
        axis=0,
    )


    confidence_upper = np.quantile(
        current_bootstrap,
        0.975,
        axis=0,
    )


    standard_errors = (
        current_bootstrap.std(
            axis=0,
            ddof=1,
        )
    )


    for metric_index, metric_name in enumerate(
        METRIC_NAMES
    ):

        size_summary_rows.append(
            {
                "n_items":
                    n_items,

                "metric":
                    metric_name,

                "estimate":
                    size_observed_metrics[
                        n_items
                    ][
                        metric_index
                    ],

                "bootstrap_standard_error":
                    standard_errors[
                        metric_index
                    ],

                "ci_95_lower":
                    confidence_lower[
                        metric_index
                    ],

                "ci_95_upper":
                    confidence_upper[
                        metric_index
                    ],

                "matched_groups":
                    100,

                "conditions":
                    4800,

                "bootstrap_replicates":
                    SENSITIVITY_BOOTSTRAP_REPLICATES,

                "resampling_unit":
                    "matched_group_within_candidate_size",
            }
        )


size_summary_df = pd.DataFrame(
    size_summary_rows
)


size_replicate_rows = []


for bootstrap_index in range(
    SENSITIVITY_BOOTSTRAP_REPLICATES
):

    for size_index, n_items in enumerate(
        CANDIDATE_SIZE_ORDER
    ):

        replicate_row = {
            "bootstrap_replicate":
                bootstrap_index,

            "n_items":
                n_items,
        }


        for metric_index, metric_name in enumerate(
            METRIC_NAMES
        ):

            replicate_row[
                metric_name
            ] = (
                size_bootstrap_values[
                    bootstrap_index,
                    size_index,
                    metric_index,
                ]
            )


        size_replicate_rows.append(
            replicate_row
        )


size_bootstrap_replicates_df = pd.DataFrame(
    size_replicate_rows
)


# ==================================================
# PART B — Weight-profile sensitivity
# ==================================================

weight_cluster_df = (
    coverage_df
    .groupby(
        [
            "n_items",
            "matched_group_id",
            "weight_profile",
        ],
        as_index=False,
    )
    .agg(
        comparable_rows=(
            "optimality_gap_percent",
            "size",
        ),

        gap_sum=(
            "optimality_gap_percent",
            "sum",
        ),

        exact_recovery_sum=(
            "exact_recovery_numeric",
            "sum",
        ),

        jaccard_sum=(
            "jaccard_similarity",
            "sum",
        ),

        large_gap_sum=(
            "large_gap_numeric",
            "sum",
        ),

        more_items_sum=(
            "more_items_numeric",
            "sum",
        ),

        selected_count_difference_sum=(
            "selected_count_difference",
            "sum",
        ),
    )
)


# Each group × profile contains:
# 3 regimes × 4 thresholds = 12 rows.
assert (
    weight_cluster_df[
        "comparable_rows"
    ]
    ==
    12
).all()


weight_arrays = {}


for n_items in CANDIDATE_SIZE_ORDER:

    weight_arrays[
        n_items
    ] = {}


    reference_group_ids = None


    for weight_profile in WEIGHT_PROFILE_ORDER:

        current_df = (
            weight_cluster_df[
                (
                    weight_cluster_df[
                        "n_items"
                    ]
                    ==
                    n_items
                )
                &
                (
                    weight_cluster_df[
                        "weight_profile"
                    ]
                    ==
                    weight_profile
                )
            ]
            .sort_values(
                "matched_group_id"
            )
            .reset_index(
                drop=True
            )
        )


        current_group_ids = (
            current_df[
                "matched_group_id"
            ].tolist()
        )


        assert len(
            current_group_ids
        ) == 100


        if reference_group_ids is None:

            reference_group_ids = (
                current_group_ids
            )

        else:

            assert (
                current_group_ids
                ==
                reference_group_ids
            )


        weight_arrays[
            n_items
        ][
            weight_profile
        ] = (
            current_df[
                AGGREGATE_COLUMNS
            ]
            .to_numpy(
                dtype=float
            )
        )


weight_observed_totals = {
    weight_profile: np.zeros(
        len(
            AGGREGATE_COLUMNS
        ),
        dtype=float,
    )
    for weight_profile in WEIGHT_PROFILE_ORDER
}


for n_items in CANDIDATE_SIZE_ORDER:

    for weight_profile in WEIGHT_PROFILE_ORDER:

        weight_observed_totals[
            weight_profile
        ] += (
            weight_arrays[
                n_items
            ][
                weight_profile
            ]
            .sum(
                axis=0
            )
        )


weight_observed_metrics = {
    weight_profile:
        calculate_metrics(
            weight_observed_totals[
                weight_profile
            ]
        )
    for weight_profile in WEIGHT_PROFILE_ORDER
}


weight_descriptive_rows = []


for weight_profile in WEIGHT_PROFILE_ORDER:

    current_metrics = (
        weight_observed_metrics[
            weight_profile
        ]
    )


    weight_descriptive_rows.append(
        {
            "weight_profile":
                weight_profile,

            "comparable_rows":
                int(
                    weight_observed_totals[
                        weight_profile
                    ][0]
                ),

            **{
                metric_name:
                    current_metrics[
                        metric_index
                    ]
                for metric_index, metric_name
                in enumerate(
                    METRIC_NAMES
                )
            },
        }
    )


weight_descriptive_df = pd.DataFrame(
    weight_descriptive_rows
)


# --------------------------------------------------
# Shared paired bootstrap across weight profiles
# --------------------------------------------------

weight_bootstrap_rng = np.random.default_rng(
    SENSITIVITY_BOOTSTRAP_SEED
    +
    10_000
)


weight_bootstrap_differences = np.empty(
    (
        SENSITIVITY_BOOTSTRAP_REPLICATES,
        len(
            WEIGHT_PROFILE_CONTRASTS
        ),
        len(
            METRIC_NAMES
        ),
    ),
    dtype=float,
)


for bootstrap_index in range(
    SENSITIVITY_BOOTSTRAP_REPLICATES
):

    bootstrap_totals = {
        weight_profile: np.zeros(
            len(
                AGGREGATE_COLUMNS
            ),
            dtype=float,
        )
        for weight_profile in WEIGHT_PROFILE_ORDER
    }


    for n_items in CANDIDATE_SIZE_ORDER:

        number_of_groups = (
            weight_arrays[
                n_items
            ][
                REFERENCE_WEIGHT_PROFILE
            ].shape[0]
        )


        sampled_indices = (
            weight_bootstrap_rng.integers(
                low=0,
                high=number_of_groups,
                size=number_of_groups,
            )
        )


        for weight_profile in WEIGHT_PROFILE_ORDER:

            bootstrap_totals[
                weight_profile
            ] += (
                weight_arrays[
                    n_items
                ][
                    weight_profile
                ][
                    sampled_indices
                ]
                .sum(
                    axis=0
                )
            )


    bootstrap_metrics = {
        weight_profile:
            calculate_metrics(
                bootstrap_totals[
                    weight_profile
                ]
            )
        for weight_profile in WEIGHT_PROFILE_ORDER
    }


    for contrast_index, (
        first_profile,
        second_profile,
    ) in enumerate(
        WEIGHT_PROFILE_CONTRASTS
    ):

        weight_bootstrap_differences[
            bootstrap_index,
            contrast_index,
        ] = (
            bootstrap_metrics[
                first_profile
            ]
            -
            bootstrap_metrics[
                second_profile
            ]
        )


weight_contrast_rows = []


for contrast_index, (
    first_profile,
    second_profile,
) in enumerate(
    WEIGHT_PROFILE_CONTRASTS
):

    observed_difference = (
        weight_observed_metrics[
            first_profile
        ]
        -
        weight_observed_metrics[
            second_profile
        ]
    )


    current_bootstrap = (
        weight_bootstrap_differences[
            :,
            contrast_index,
            :,
        ]
    )


    confidence_lower = np.quantile(
        current_bootstrap,
        0.025,
        axis=0,
    )


    confidence_upper = np.quantile(
        current_bootstrap,
        0.975,
        axis=0,
    )


    standard_errors = (
        current_bootstrap.std(
            axis=0,
            ddof=1,
        )
    )


    for metric_index, metric_name in enumerate(
        METRIC_NAMES
    ):

        weight_contrast_rows.append(
            {
                "first_profile":
                    first_profile,

                "second_profile":
                    second_profile,

                "contrast":
                    (
                        f"{first_profile}"
                        f" minus "
                        f"{second_profile}"
                    ),

                "metric":
                    metric_name,

                "estimated_difference":
                    observed_difference[
                        metric_index
                    ],

                "bootstrap_standard_error":
                    standard_errors[
                        metric_index
                    ],

                "ci_95_lower":
                    confidence_lower[
                        metric_index
                    ],

                "ci_95_upper":
                    confidence_upper[
                        metric_index
                    ],

                "ci_excludes_zero":
                    (
                        confidence_lower[
                            metric_index
                        ]
                        >
                        0.0
                    )
                    or
                    (
                        confidence_upper[
                            metric_index
                        ]
                        <
                        0.0
                    ),

                "matched_groups":
                    500,

                "bootstrap_replicates":
                    SENSITIVITY_BOOTSTRAP_REPLICATES,

                "resampling_design":
                    (
                        "shared_across_weight_profiles_"
                        "stratified_by_candidate_size"
                    ),

                "multiple_comparison_adjustment":
                    "none",
            }
        )


weight_contrast_df = pd.DataFrame(
    weight_contrast_rows
)


weight_replicate_rows = []


for bootstrap_index in range(
    SENSITIVITY_BOOTSTRAP_REPLICATES
):

    for contrast_index, (
        first_profile,
        second_profile,
    ) in enumerate(
        WEIGHT_PROFILE_CONTRASTS
    ):

        replicate_row = {
            "bootstrap_replicate":
                bootstrap_index,

            "first_profile":
                first_profile,

            "second_profile":
                second_profile,

            "contrast":
                (
                    f"{first_profile}"
                    f" minus "
                    f"{second_profile}"
                ),
        }


        for metric_index, metric_name in enumerate(
            METRIC_NAMES
        ):

            replicate_row[
                metric_name
            ] = (
                weight_bootstrap_differences[
                    bootstrap_index,
                    contrast_index,
                    metric_index,
                ]
            )


        weight_replicate_rows.append(
            replicate_row
        )


weight_bootstrap_replicates_df = pd.DataFrame(
    weight_replicate_rows
)


# --------------------------------------------------
# Audit
# --------------------------------------------------

size_group_counts = (
    coverage_df
    .groupby(
        "matched_group_id"
    )
    .size()
)


weight_group_counts = (
    coverage_df
    .groupby(
        [
            "matched_group_id",
            "weight_profile",
        ]
    )
    .size()
)


sensitivity_audit_df = pd.DataFrame(
    [
        {
            "benchmark_rows":
                len(
                    coverage_df
                ),

            "matched_groups":
                coverage_df[
                    "matched_group_id"
                ].nunique(),

            "candidate_sizes":
                coverage_df[
                    "n_items"
                ].nunique(),

            "weight_profiles":
                coverage_df[
                    "weight_profile"
                ].nunique(),

            "minimum_rows_per_matched_group":
                int(
                    size_group_counts.min()
                ),

            "maximum_rows_per_matched_group":
                int(
                    size_group_counts.max()
                ),

            "minimum_rows_per_group_profile":
                int(
                    weight_group_counts.min()
                ),

            "maximum_rows_per_group_profile":
                int(
                    weight_group_counts.max()
                ),

            "negative_gap_rows":
                int(
                    (
                        coverage_df[
                            "optimality_gap_percent"
                        ]
                        <
                        -1e-8
                    ).sum()
                ),

            "bootstrap_replicates":
                SENSITIVITY_BOOTSTRAP_REPLICATES,

            "candidate_size_design":
                (
                    "independent generated groups "
                    "within each size"
                ),

            "weight_profile_design":
                (
                    "paired within matched generated groups"
                ),

            "confidence_intervals":
                "unadjusted_percentile_95",
        }
    ]
)


assert (
    size_group_counts
    ==
    48
).all()


assert (
    weight_group_counts
    ==
    12
).all()


assert sensitivity_audit_df.loc[
    0,
    "negative_gap_rows",
] == 0


# --------------------------------------------------
# Save new Step-133 outputs
# --------------------------------------------------

size_summary_df.to_csv(
    SIZE_SUMMARY_PATH,
    index=False,
)


size_bootstrap_replicates_df.to_csv(
    SIZE_BOOTSTRAP_REPLICATES_PATH,
    index=False,
)


weight_descriptive_df.to_csv(
    WEIGHT_DESCRIPTIVE_PATH,
    index=False,
)


weight_contrast_df.to_csv(
    WEIGHT_CONTRAST_PATH,
    index=False,
)


weight_bootstrap_replicates_df.to_csv(
    WEIGHT_BOOTSTRAP_REPLICATES_PATH,
    index=False,
)


sensitivity_audit_df.to_csv(
    SENSITIVITY_AUDIT_PATH,
    index=False,
)


# --------------------------------------------------
# Readable displays
# --------------------------------------------------

size_display_df = size_summary_df.copy()


size_display_df[
    "estimate_and_ci"
] = (
    size_display_df[
        "estimate"
    ].map(
        lambda value:
            f"{value:.4f}"
    )
    +
    " ["
    +
    size_display_df[
        "ci_95_lower"
    ].map(
        lambda value:
            f"{value:.4f}"
    )
    +
    ", "
    +
    size_display_df[
        "ci_95_upper"
    ].map(
        lambda value:
            f"{value:.4f}"
    )
    +
    "]"
)


weight_display_df = weight_contrast_df.copy()


weight_display_df[
    "difference_and_ci"
] = (
    weight_display_df[
        "estimated_difference"
    ].map(
        lambda value:
            f"{value:.4f}"
    )
    +
    " ["
    +
    weight_display_df[
        "ci_95_lower"
    ].map(
        lambda value:
            f"{value:.4f}"
    )
    +
    ", "
    +
    weight_display_df[
        "ci_95_upper"
    ].map(
        lambda value:
            f"{value:.4f}"
    )
    +
    "]"
)


print(
    "======================================"
)

print(
    "COVERAGE SIZE/WEIGHT SENSITIVITY AUDIT"
)

print(
    "======================================"
)


print(
    sensitivity_audit_df.to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "COVERAGE RESULTS BY CANDIDATE-SET SIZE"
)

print(
    "======================================"
)


print(
    size_display_df[
        [
            "n_items",
            "metric",
            "estimate_and_ci",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n======================================"
)

print(
    "COVERAGE RESULTS BY WEIGHT PROFILE"
)

print(
    "======================================"
)


print(
    weight_descriptive_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
    )
)


print(
    "\n======================================"
)

print(
    "PAIRED WEIGHT-PROFILE CONTRASTS"
)

print(
    "======================================"
)


print(
    weight_display_df[
        [
            "first_profile",
            "second_profile",
            "metric",
            "difference_and_ci",
            "ci_excludes_zero",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nCandidate-size summary saved to:"
)

print(
    SIZE_SUMMARY_PATH
)


print(
    "\nWeight-profile descriptive summary saved to:"
)

print(
    WEIGHT_DESCRIPTIVE_PATH
)


print(
    "\nPaired weight-profile contrasts saved to:"
)

print(
    WEIGHT_CONTRAST_PATH
)


print(
    "\nSensitivity audit saved to:"
)

print(
    SENSITIVITY_AUDIT_PATH
)
