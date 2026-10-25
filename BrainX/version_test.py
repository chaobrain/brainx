# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-


import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

import BrainX

# Repository root: BrainX/version_test.py -> parents[1].
_REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"

# Matches an exact pin, e.g. ``brainstate==0.5.1`` (ignores ``>=``/``<=`` ranges,
# comments, blank lines, and any trailing environment markers).
_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;#]+)")


def _parse_pinned_requirements():
    """Return the ``name == version`` pins declared in ``requirements.txt``."""
    pins = {}
    for raw in _REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN_RE.match(line)
        if match:
            pins[match.group(1)] = match.group(2)
    return pins


_PINNED = _parse_pinned_requirements()


class Test:
    def test(self):
        print(BrainX.__version__)

    def test_imports(self):
        import brainmass
        import brainpy.state
        import brainspike
        import braincell
        import braintools
        import braintrace
        import brainevent
        import brainunit
        import brainstate

        print(brainmass.__version__)
        print(brainpy.__version__)
        print(brainspike.__version__)
        print(braincell.__version__)
        print(braintools.__version__)
        print(braintrace.__version__)
        print(brainevent.__version__)
        print(brainunit.__version__)
        print(brainstate.__version__)

    def test_requirements_are_pinned(self):
        """Every brain* ecosystem package must be exactly pinned."""
        expected = {
            "brainunit",
            "brainevent",
            "brainstate",
            "braintools",
            "braintrace",
            "braincell",
            "brainpy",
            "brainspikes",
            "brainmass",
        }
        missing = sorted(expected - set(_PINNED))
        assert not missing, f"requirements.txt is missing exact pins for: {missing}"

    @pytest.mark.parametrize("dist, pinned", sorted(_PINNED.items()))
    def test_installed_version_matches_requirements(self, dist, pinned):
        """The installed distribution version must equal the requirements.txt pin."""
        try:
            installed = version(dist)
        except PackageNotFoundError:
            pytest.skip(f"{dist} is not installed in this environment")
        assert installed == pinned, (
            f"{dist}: installed {installed!r} != requirements.txt pin {pinned!r}"
        )
