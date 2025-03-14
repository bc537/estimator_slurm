# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Wrapper class of scipy.optimize.minimize."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any
from scipy.optimize import basinhopping

from qiskit_algorithms.utils.validation import validate_min
from qiskit_algorithms.optimizers.optimizer import Optimizer, POINT
from qiskit_algorithms.optimizers import OptimizerResult


class BasinHopping(Optimizer):

    def __init__(
            self,
            options: dict[str, Any] | None = None,
            minimizer_kwargs: dict[str, Any] | None = None,
            max_evals_grouped: int = 1,
    ):
        """
        Args:
            options: A dictionary of solver options. Defaults:
                    niter=100, T=1.0, stepsize=0.5,
                     take_step=None, accept_test=None,
                     callback=None, interval=50, disp=False, niter_success=None,
                     seed=None, target_accept_rate=0.5, stepwise_factor=0.9

            kwargs: additional kwargs for scipy.optimize.minimize.
            max_evals_grouped: Max number of default gradient evaluations performed simultaneously.
        """

        _default_options = dict(niter=100, T=1.0, stepsize=0.5,
                                take_step=None, accept_test=None,
                                callback=None, interval=50, disp=False, niter_success=None,
                                seed=None, target_accept_rate=0.5, stepwise_factor=0.9)

        if options is None:
            self._options = _default_options
        else:
            self._options = options
            for key in _default_options:
                if key not in self._options:
                    self._options[key] = _default_options[key]

        validate_min("max_evals_grouped", max_evals_grouped, 1)
        self._minimizer_kwargs = minimizer_kwargs
        self._max_evals_grouped = max_evals_grouped

    @property
    def settings(self) -> dict[str, Any]:
        options = self._options.copy()
        if hasattr(self, "_OPTIONS"):
            # all _OPTIONS should be keys in self._options, but add a failsafe here
            attributes = [
                option
                for option in self._OPTIONS  # pylint: disable=no-member
                if option in options.keys()
            ]

            settings = {attr: options.pop(attr) for attr in attributes}
        else:
            settings = {}

        settings["max_evals_grouped"] = self._max_evals_grouped
        settings["options"] = options

        return settings

    def get_support_level(self):
        pass

    def minimize(
            self,
            fun: Callable[[POINT], float],
            x0: POINT,
            jac: Callable[[POINT], POINT] | None = None,
            bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:

        # These are not used here
        jac = None
        bounds = None

        raw_result = basinhopping(fun, x0, minimizer_kwargs=self._minimizer_kwargs, **self._options)

        result = OptimizerResult()
        result.x = raw_result.x
        result.fun = raw_result.fun
        result.nfev = raw_result.nfev
        result.njev = raw_result.get("njev", None)
        result.nit = raw_result.get("nit", None)

        return result
