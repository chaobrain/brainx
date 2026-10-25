# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Cross-package compatibility and basic-correctness tests for the BrainX stack.

Where ``version_test.py`` only checks that the pinned versions are coherent, this
module exercises *interoperation*: the pinned ecosystem packages are imported
together and driven through small, deterministic computations that cut across
package boundaries (units in state + transforms, event/sparse operators, the
``braintrace`` ↔ ``brainstate`` eligibility-trace path, neural-mass and
single-compartment integration, …).

The assertions are intentionally *hard* rather than skipped on error: the whole
point of the suite is to surface a real incompatibility in the pinned set, so a
genuine break must fail loudly. Numerical checks use exact discrete references or
loose tolerances so they stay valid across the supported JAX-version matrix.

All repeated dynamics are driven with ``brainstate.transform`` primitives
(``for_loop`` / ``jit``), never bare Python loops.
"""

import re

import jax.numpy as jnp
import numpy as np

import brainstate
import brainunit as u


def _version_tuple(text):
    """Parse a leading ``MAJOR.MINOR.PATCH`` into a comparable int tuple."""
    parts = re.findall(r"\d+", text)
    return tuple(int(p) for p in parts[:3])


# ---------------------------------------------------------------------------
# Version coherence — the coupled brainstate>=0.5.2 / braintrace>=0.2.2 contract.
# ---------------------------------------------------------------------------

def test_coupled_versions_are_coherent():
    """braintrace 0.2.2 requires brainstate>=0.5.2; the pins must reflect that."""
    import braintrace

    assert _version_tuple(brainstate.__version__) >= (0, 5, 2), brainstate.__version__
    assert _version_tuple(braintrace.__version__) >= (0, 2, 2), braintrace.__version__


def test_new_feature_contract_present():
    """The APIs that couple the two bumps are importable/callable."""
    import braintrace

    # Added in brainstate 0.5.2 — the probe predicate braintrace.compile relies on.
    assert hasattr(brainstate.transform, "in_new_state_probe")
    assert callable(brainstate.transform.in_new_state_probe)
    # The unified entry point introduced/cemented in braintrace 0.2.2.
    assert callable(braintrace.compile)


# ---------------------------------------------------------------------------
# brainunit ↔ brainstate — unit-carrying State integrated by a transform.
# ---------------------------------------------------------------------------

def test_brainunit_state_decay_under_for_loop():
    """A unit-aware State decays correctly under ``transform.for_loop``.

    Forward-Euler ``dV = -V/tau * dt`` for ``n`` steps has the exact closed form
    ``V0 * (1 - dt/tau)**n``; with ``T = n*dt = tau`` it also approximates the
    continuous ``V0 * exp(-1)``. Both the magnitude and the ``mV`` unit must
    survive the transform.
    """
    tau, dt, v0, n = 10.0 * u.ms, 0.1 * u.ms, 1.0 * u.mV, 100
    v = brainstate.HiddenState(v0)

    def step(_):
        v.value = v.value - v.value / tau * dt
        return v.value

    brainstate.transform.for_loop(step, np.arange(n))
    final = v.value

    ratio = float(dt / tau)  # dimensionless 0.01
    exact = v0 * (1.0 - ratio) ** n
    assert u.math.allclose(final, exact, rtol=1e-4), (final, exact)

    # ``to_decimal(mV)`` raises unless ``final`` is voltage-dimensioned, so this
    # both checks the value and pins the unit.
    final_mv = final.to_decimal(u.mV)
    assert abs(final_mv - float(np.exp(-1))) < 0.05, final_mv


# ---------------------------------------------------------------------------
# brainevent — event-driven / sparse operators agree with dense references.
# ---------------------------------------------------------------------------

def test_brainevent_binary_matmul_matches_dense():
    """A spike (binary) vector through brainevent equals the dense matmul.

    brainevent's CPU event/sparse kernels are generated with numba, which the
    test environment provides via the ``brainx[cpu]`` extra.
    """
    import brainevent

    spikes = jnp.asarray([1, 0, 1, 0, 1], dtype=bool)
    weight = jnp.asarray(np.random.RandomState(0).rand(5, 3))

    out = jnp.asarray(brainevent.BinaryArray(spikes) @ weight)
    ref = spikes.astype(weight.dtype) @ weight
    assert out.shape == (3,)
    assert jnp.allclose(out, ref), (out, ref)


def test_brainevent_csr_matches_dense():
    """A brainevent CSR matrix-vector product equals its dense reference."""
    import brainevent

    dense = jnp.asarray(np.random.RandomState(1).rand(4, 5))
    vector = jnp.asarray(np.random.RandomState(2).rand(5))

    csr = brainevent.CSR.fromdense(dense)
    assert jnp.allclose(jnp.asarray(csr @ vector), dense @ vector)


# ---------------------------------------------------------------------------
# braintools ↔ brainstate — initializers, metrics, and an nn layer.
# ---------------------------------------------------------------------------

def test_braintools_init_and_metric_are_correct():
    """A deterministic initializer and metric give exactly the expected values."""
    import braintools

    weight = braintools.init.Constant(0.5)([2, 3])
    assert weight.shape == (2, 3)
    assert jnp.allclose(jnp.asarray(weight), 0.5)

    pred = jnp.asarray([1.0, 2.0, 3.0])
    target = jnp.asarray([1.0, 2.0, 5.0])
    # |pred - target| = [0, 0, 2]  ->  mean 2/3.
    mae = float(jnp.mean(jnp.asarray(braintools.metric.absolute_error(pred, target))))
    assert abs(mae - 2.0 / 3.0) < 1e-6, mae


def test_brainstate_nn_linear_forward():
    """A ``brainstate.nn`` layer runs and produces a finite, correctly shaped output."""
    layer = brainstate.nn.Linear(4, 3)
    out = layer(jnp.ones((2, 4)))
    assert out.shape == (2, 3)
    assert bool(jnp.all(jnp.isfinite(out)))


# ---------------------------------------------------------------------------
# brainpy.state — a point neuron steps under a dt context (reuses brainstate).
# ---------------------------------------------------------------------------

def test_brainpy_state_neuron_step():
    """A ``brainpy.state`` LIF neuron initialises and advances one finite step."""
    import brainpy

    with brainstate.environ.context(dt=0.1 * u.ms):
        neuron = brainpy.state.LIF(5)
        brainstate.nn.init_all_states(neuron)
        out = neuron(jnp.ones(5) * u.mA)

    assert jnp.asarray(out).shape == (5,)
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(out))))


# ---------------------------------------------------------------------------
# brainspike — a NEST-compatible neuron runs under the Simulator.
# ---------------------------------------------------------------------------

def test_brainspike_simulator_run():
    """A ``brainspike`` ``iaf_psc_alpha`` neuron simulates and records a finite trace."""
    import brainspike

    sim = brainspike.Simulator(dt=0.1 * u.ms)
    neuron = sim.create(brainspike.iaf_psc_alpha, 1, I_e=376.0 * u.pA)
    vm = sim.create(brainspike.voltmeter)
    sim.connect(vm, neuron)
    res = sim.simulate(10.0 * u.ms)
    v = jnp.asarray(u.get_mantissa(res.trace(vm, "V_m") / u.mV))

    assert v.shape[1] == 1
    assert bool(jnp.all(jnp.isfinite(v)))


# ---------------------------------------------------------------------------
# braincell — a single compartment integrates to a finite membrane voltage.
# ---------------------------------------------------------------------------

def test_braincell_single_compartment_integration():
    """A ``braincell.SingleCompartment`` integrates stably under ``for_loop``.

    Driving a default compartment with a constant current *density* and stepping
    it through ``brainstate.transform.for_loop`` must leave the membrane voltage
    finite and in millivolts (exercising braincell ↔ brainstate ↔ brainunit).
    """
    import braincell

    with brainstate.environ.context(dt=0.01 * u.ms):
        cell = braincell.SingleCompartment(size=1)
        brainstate.nn.init_all_states(cell)

        def step(_):
            cell.update(1.0 * u.uA / u.cm ** 2)
            return cell.V.value

        voltages = brainstate.transform.for_loop(step, np.arange(50))

    millivolts = voltages.to_decimal(u.mV)  # raises unless mV-dimensioned
    assert millivolts.shape == (50, 1)
    assert bool(jnp.all(jnp.isfinite(millivolts)))


# ---------------------------------------------------------------------------
# brainmass — a neural-mass model simulates to a finite trajectory.
# ---------------------------------------------------------------------------

def test_brainmass_meanfield_run():
    """A ``brainmass`` mean-field node runs through the Simulator to finite output."""
    import brainmass

    assert len(brainmass.list_models()) == 20

    node = brainmass.HopfStep(2, a=-0.2)
    sim = brainmass.Simulator(node, dt=0.1 * u.ms)
    result = sim.run(10.0 * u.ms, monitors=["x"])

    trajectory = np.asarray(result["x"])
    assert trajectory.shape == (100, 2)
    assert np.all(np.isfinite(trajectory))


# ---------------------------------------------------------------------------
# braintrace ↔ brainstate — the eligibility-trace compile path (headline interop).
# ---------------------------------------------------------------------------

def test_braintrace_compile_etrace_learner():
    """``braintrace.compile`` builds and runs a tiny online learner on brainstate.

    This exercises the probe-deferral path: ``compile`` runs the eager
    ``vmap_new_states`` discovery probe (whose cooperation predicate,
    ``in_new_state_probe``, is the brainstate 0.5.2 feature) before producing the
    real eligibility-trace algorithm.
    """
    import braintrace

    class SimpleGRU(brainstate.nn.Module):
        def __init__(self, n_in, n_rec, n_out):
            super().__init__()
            self.rnn = braintrace.nn.GRUCell(n_in, n_rec)
            self.out = braintrace.nn.Linear(n_rec, n_out)

        def update(self, x):
            return self.out(self.rnn(x))

    model = SimpleGRU(8, 16, 4)
    learner = braintrace.compile(model, braintrace.D_RTRL, jnp.zeros(8))

    out = learner(jnp.ones(8))
    assert jnp.asarray(out).shape == (4,)
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(out))))
