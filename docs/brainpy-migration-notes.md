# BrainPy migration notes

Practical notes for moving a [`brainpy`][brainpy] project onto the
[`brainx`][brainx] packages: what maps to what, a worked before/after example, and the
cases that do not carry over.

For why the ecosystem is arranged this way, which package owns which modeling scale,
and whether your project should move at all, read
[BrainPy to BrainX](./brainpy-to-brainx.md) first.

## The two packages at a glance

`brainpy` and `brainpy.state` ship together (BrainPy >= 2.9.0) but share little else:

| | `brainpy` | `brainpy.state` |
| --- | --- | --- |
| import | `import brainpy as bp` | `import brainpy.state` |
| state objects | `brainpy.math.Variable` | `brainstate.State` |
| module base class | `brainpy.DynamicalSystem` | `brainstate.nn.Module` |
| simulation loop | `brainpy.DSRunner` | `brainstate.transform.for_loop` |
| parameters | plain floats (ms, mV by convention) | [`brainunit`][brainunit] quantities |

Both are installed by `pip install -U BrainX`, or with `pip install "brainpy>=2.9.0"`.
They import cleanly into the same process, so a
project can migrate one model at a time.

## Migration in four steps

Migration is incremental. In order of effort and payoff:

1. **Port the model definition.** `brainpy.DynamicalSystem` → `brainstate.nn.Module`,
   `brainpy.dyn.*` → `brainpy.state.*`. Add units to parameters as you go.
2. **Replace the runner.** `brainpy.DSRunner` / `brainpy.LoopOverTime` →
   `brainstate.transform.for_loop`. Never drive the model with a bare Python loop:
   `for_loop` traces the body once and compiles the whole rollout into a single XLA
   program.
3. **Swap the helper modules** — losses, initializers, inputs, connectivity and plots all
   move to [`braintools`][braintools] (table below). These are mostly mechanical renames.
4. **Leave `brainpy.analysis` where it is.** Keep a reduced `brainpy` model alongside the
   migrated one if you need phase-plane or bifurcation analysis.

## Worked example: an E/I network

The same conductance-based E/I network, before and after. Both versions run against the
pinned releases and produce comparable firing statistics.

**Before — `brainpy`:**

```python
import brainpy as bp
import brainpy.math as bm

bm.set_dt(0.1)


class EINet(bp.DynSysGroup):
    def __init__(self, n_exc=320, n_inh=80, prob=0.02):
        super().__init__()
        num = n_exc + n_inh
        self.n_exc = n_exc
        self.N = bp.dyn.LifRef(
            num, V_rest=-60., V_th=-50., V_reset=-60.,
            tau=20., tau_ref=5.,
            V_initializer=bp.init.Normal(-55., 2.),
        )
        self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
        self.E = bp.dyn.HalfProjAlignPostMg(
            comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=n_exc, post=num), weight=0.6),
            syn=bp.dyn.Expon.desc(size=num, tau=5.),
            out=bp.dyn.COBA.desc(E=0.),
            post=self.N,
        )
        self.I = bp.dyn.HalfProjAlignPostMg(
            comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=n_inh, post=num), weight=6.7),
            syn=bp.dyn.Expon.desc(size=num, tau=10.),
            out=bp.dyn.COBA.desc(E=-80.),
            post=self.N,
        )

    def update(self, inp=20.):
        spk = self.delay.at('I')
        self.E(spk[:self.n_exc])
        self.I(spk[self.n_exc:])
        self.delay(self.N(inp))
        return self.N.spike.value


net = EINet()
runner = bp.DSRunner(net, monitors=['N.spike'])
runner.run(100.)
spikes = runner.mon['N.spike']
```

**After — `brainpy.state`:**

```python
import brainpy.state
import brainstate
import braintools
import brainunit as u

brainstate.environ.set(dt=0.1 * u.ms)


class EINet(brainstate.nn.Module):
    def __init__(self, n_exc=320, n_inh=80, prob=0.02):
        super().__init__()
        num = n_exc + n_inh
        self.n_exc = n_exc
        self.N = brainpy.state.LIFRef(
            num, V_rest=-60. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
            tau=20. * u.ms, tau_ref=5. * u.ms,
            V_initializer=braintools.init.Normal(-55. * u.mV, 2. * u.mV),
        )
        self.E = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(n_exc, num, prob, 0.6 * u.mS),
            syn=brainpy.state.Expon.desc(num, tau=5. * u.ms),
            out=brainpy.state.COBA.desc(E=0. * u.mV),
            post=self.N,
        )
        self.I = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(n_inh, num, prob, 6.7 * u.mS),
            syn=brainpy.state.Expon.desc(num, tau=10. * u.ms),
            out=brainpy.state.COBA.desc(E=-80. * u.mV),
            post=self.N,
        )

    def update(self, t, inp):
        with brainstate.environ.context(t=t):
            spk = self.N.get_spike() != 0.
            self.E(spk[:self.n_exc])
            self.I(spk[self.n_exc:])
            self.N(inp)
            return self.N.get_spike()


net = EINet()
brainstate.nn.init_all_states(net)

times = u.math.arange(0. * u.ms, 100. * u.ms, brainstate.environ.get_dt())
spikes = brainstate.transform.for_loop(lambda t: net.update(t, 20. * u.mA), times)
```

Four differences carry most of the work: parameters carry units, states are initialized
explicitly with `init_all_states`, the delay bookkeeping moves into the projection
instead of an explicit `VarDelay`, and the runner is replaced by a compiled `for_loop`.

## Runtime mapping

| `brainpy` | [`brainx`][brainx] |
| --- | --- |
| `brainpy.math.Variable` | `brainstate.HiddenState` / `brainstate.ParamState` |
| `brainpy.DynamicalSystem`, `brainpy.DynSysGroup` | `brainstate.nn.Module` |
| `brainpy.DSRunner`, `brainpy.LoopOverTime` | `brainstate.transform.for_loop` / `scan` |
| `brainpy.math.jit` | `brainstate.transform.jit` |
| `brainpy.math.random` | `brainstate.random` |
| `brainpy.math.set_dt`, `brainpy.share` | `brainstate.environ` |
| `brainpy.BPTT` and the trainers | `brainstate.transform` gradients + `braintools.optim` |
| long rollouts under autograd | `brainstate.transform.checkpointed_for_loop` |

## Module mapping

| `brainpy` | [`brainx`][brainx] |
| --- | --- |
| `brainpy.dyn` (point neurons, synapses, projections) | `brainpy.state` |
| `brainpy.dyn` channels, ions, `CondNeuGroup` | [`braincell`][braincell] |
| `brainpy.rates`, `brainpy.dyn` rate models | [`brainmass`][brainmass] |
| `brainpy.dnn`, `brainpy.layers` | `brainstate.nn` |
| `brainpy.losses`, `brainpy.measure` | `braintools.metric` |
| `brainpy.initialize` | `braintools.init` |
| `brainpy.inputs` | `braintools.input` |
| `brainpy.conn` | `braintools.conn` |
| `brainpy.optim` | `braintools.optim` |
| `brainpy.visualization` | `braintools.visualize` |
| `brainpy.encoding` | `braintools` encoders |
| `brainpy.math.surrogate` | `braintools.surrogate` |
| `brainpy.math.sparse`, `.event`, `.jitconn` | [`brainevent`][brainevent] |
| `brainpy.odeint`, `brainpy.sdeint` | `braintools.quad`, `brainstate.nn.exp_euler_step` |
| `brainpy.analysis` | *no replacement — see below* |

## What does not carry over

**`brainunit` quantities.** `brainpy` depends on [`brainunit`][brainunit] internally, but
its array layer does not accept quantities:

```python
>>> import brainpy.math as bm, brainunit as u
>>> bm.exp(10. * u.mV)
TypeError: exp requires ndarray or scalar arguments, got <class 'saiunit.Quantity'> at position 0.
```

Strip units at the boundary — `(10. * u.mV).to_decimal(u.mV)` — or migrate the model.
End-to-end unit safety is only available on the [`brainx`][brainx] side.

**Online learning.** [`braintrace`][braintrace] attaches eligibility traces to operations
inside `brainstate`-based modules. It cannot see a `brainpy.math.Variable` graph, so
`brainpy` models cannot be trained online with it.

**Mixed model trees.** Modules from the two state systems cannot be composed into one
model. A `brainpy.state` module placed inside a `brainpy.DynSysGroup` never gets its
states initialized, and fails at run time:

```python
AttributeError: 'LIF' object has no attribute 'V'
```

Port whole models, not individual layers. The two can still live in the same script as
separate models.

**Analysis.** `brainpy.analysis` — phase-plane analyzers, `Bifurcation1D` /
`Bifurcation2D`, slow-fast decomposition — has no counterpart in [`brainx`][brainx]. The
usual pattern is to keep a reduced `brainpy` model for analysis alongside the migrated
simulation model.

## Compatibility matrix

Which ecosystem packages a `brainpy` project can use, and which can be built into the
same model:

| package | usable alongside `brainpy` | composable into one `brainpy` model |
| --- | --- | --- |
| [`brainstate`][brainstate] | yes — `brainpy` is built on it | — |
| [`brainevent`][brainevent] | yes — `brainpy` operators delegate to it | — |
| [`braintools`][braintools] | yes — `brainpy` helpers delegate to it | — |
| [`braincell`][braincell] | yes, as a separate model | no |
| [`brainmass`][brainmass] | yes, as a separate model | no |
| [`brainunit`][brainunit] | only outside `brainpy.math` | no |
| [`braintrace`][braintrace] | no | no |

## Where to go next

For a hands-on introduction to the target APIs, work through the
[point-neuron network tutorial](./tutorials/brainpy_point_neuron_networks.ipynb).

[brainx]: https://brainx.chaobrain.com/
[brainpy]: https://brainpy.readthedocs.io/
[brainpy.state]: https://brainpy.readthedocs.io/state/index.html
[brainstate]: https://brainx.chaobrain.com/brainstate/
[brainevent]: https://brainx.chaobrain.com/brainevent/
[braintools]: https://brainx.chaobrain.com/braintools/
[braincell]: https://brainx.chaobrain.com/braincell/
[brainmass]: https://brainx.chaobrain.com/brainmass/
[brainunit]: https://brainx.chaobrain.com/brainunit/
[braintrace]: https://brainx.chaobrain.com/braintrace/
