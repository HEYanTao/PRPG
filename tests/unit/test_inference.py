from __future__ import annotations

import numpy as np
import pytest

from prpg.errors import ModelError
from prpg.model.inference import (
    clopper_pearson_interval,
    higher_quantile,
    holm_correction,
    plus_one_upper_tail_pvalue,
)


def test_holm_uses_stable_step_down_and_adjusted_values() -> None:
    result = holm_correction([0.01, 0.04, 0.01, 0.20], alpha=0.05)

    np.testing.assert_array_equal(result.rejected, [True, False, True, False])
    np.testing.assert_allclose(result.adjusted_p_values, [0.04, 0.08, 0.04, 0.20])
    assert not result.rejected.flags.writeable


def test_holm_boundary_is_inclusive_and_stops_after_first_failure() -> None:
    result = holm_correction([0.025, 0.026], alpha=0.05)
    np.testing.assert_array_equal(result.rejected, [True, True])
    stopped = holm_correction([0.03, 0.04], alpha=0.05)
    np.testing.assert_array_equal(stopped.rejected, [False, False])


def test_plus_one_pvalue_counts_ties_and_higher_quantile_is_frozen() -> None:
    assert plus_one_upper_tail_pvalue(2.0, [1.0, 2.0, 3.0]) == 0.75
    assert higher_quantile([1.0, 2.0, 3.0, 4.0], 0.50) == 3.0


def test_clopper_pearson_boundaries_and_50000_precision() -> None:
    empty = clopper_pearson_interval(0, 50_000)
    full = clopper_pearson_interval(50_000, 50_000)
    middle = clopper_pearson_interval(24_834, 50_000)
    assert empty.lower == 0.0
    assert full.upper == 1.0
    assert middle.conservative_half_width == pytest.approx(0.004392602182149319)
    assert middle.conservative_half_width <= 0.005


@pytest.mark.parametrize(
    "call",
    [
        lambda: holm_correction([]),
        lambda: holm_correction([np.nan]),
        lambda: holm_correction([0.1], alpha=1.0),
        lambda: plus_one_upper_tail_pvalue(np.inf, [1.0]),
        lambda: plus_one_upper_tail_pvalue(1.0, []),
        lambda: higher_quantile([], 0.5),
        lambda: higher_quantile([1.0], 1.1),
        lambda: clopper_pearson_interval(-1, 10),
        lambda: clopper_pearson_interval(11, 10),
        lambda: clopper_pearson_interval(0, 0),
    ],
)
def test_inference_inputs_fail_closed(call: object) -> None:
    with pytest.raises(ModelError):
        call()  # type: ignore[operator]
