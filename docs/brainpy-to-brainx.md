# BrainPy in BrainX

[`brainpy`][brainpy] is the general-purpose brain dynamics programming package of
[`brainx`][brainx]. This page explains where it sits among the other packages, which
package fits which kind of model, and what an existing `brainpy` project can and cannot
combine with the rest of the ecosystem.

It describes `brainpy` 2.x and the component versions pinned by the current
[`brainx`][brainx] release; see the [change log](./CHANGELOG.md) for the exact pins.

## Two APIs in one package

Since BrainPy 2.9.0 the `brainpy` distribution ships two APIs built on different state
systems:

| | `brainpy` | `brainpy.state` |
| --- | --- | --- |
| import | `import brainpy as bp` | `import brainpy.state` |
| state objects | `brainpy.math.Variable` | `brainstate.State` |
| module base class | `brainpy.DynamicalSystem` | `brainstate.nn.Module` |
| simulation loop | `brainpy.DSRunner` | `brainstate.transform.for_loop` |
| parameters | plain floats (ms, mV by convention) | [`brainunit`][brainunit] quantities |

[`brainpy.state`][brainpy.state] was created to connect `brainpy` with the
[`brainx`][brainx] ecosystem. Its models are [`brainstate`][brainstate] modules, the same
state runtime as the other modeling packages, so they work with the whole ecosystem: they
take [`brainunit`][brainunit] quantities as parameters, they can be trained online with
[`braintrace`][braintrace], and they compose with [`braincell`][braincell] and
[`brainmass`][brainmass] models. The rest of `brainpy` predates the ecosystem and keeps
its own state system.

Both APIs are installed by `pip install -U BrainX` and import cleanly into the same
process, so a project can move one model at a time.

## Choosing a package

| your situation | recommendation |
| --- | --- |
| Maintaining a working `brainpy` project | keep it; see *Should you port?* below |
| Modeling large-scale point-neuron spiking neural networks | [`brainspike`][brainspike] |
| Modeling ions, ion channels, or morphology | [`braincell`][braincell] |
| Modeling neural-mass or whole-brain dynamics | [`brainmass`][brainmass] |
| Phase-plane or bifurcation analysis | `brainpy.analysis` |

**Large-scale point-neuron networks.** [`brainspike`][brainspike] is the dedicated package
for large-scale point-neuron spiking networks: NEST-compatible models that run on
accelerators and can be trained. Point-neuron modeling was what `brainpy` did best;
`brainpy.state` carries its model names and projection patterns onto the `brainstate`
runtime, which makes it the natural target when porting an existing `brainpy` model.

**Cells, ions, and morphology.** The ion and channel APIs in `brainpy` have known design
limitations, and its compartmental support is restricted to single-compartment models.
[`braincell`][braincell] provides comprehensive conductance-based and Hodgkin-Huxley
models together with morphologically structured, multicompartment cells. Migrate rather
than mix.

**Neural-mass and whole-brain models.** The `brainpy` rate models — FitzHugh-Nagumo,
Stuart-Landau, threshold-linear, Wilson-Cowan — work, and [`brainmass`][brainmass] has
direct counterparts. It adds models `brainpy` never had (Jansen-Rit, Epileptor, Hopf,
Montbrió-Pazó-Roxin, Wong-Wang, Larter-Breakspear) and the infrastructure whole-brain
work needs: explicit coupling schemes, structured noise processes, forward models for
BOLD, EEG, and MEG signals, and parameter fitting against empirical data.

**Dynamical-systems analysis.** `brainpy.analysis` — phase-plane analyzers,
`Bifurcation1D` / `Bifurcation2D`, slow-fast decomposition — has no counterpart in the
other packages.

## Shared foundations

The current `brainpy` codebase is built on [`brainstate`][brainstate],
[`brainevent`][brainevent], and [`braintools`][braintools]. Many internal functions
delegate to their implementations, so the two sides share the same code rather than
maintaining parallel ports:

- State handling, environment settings, and compiled transformations come from
  `brainstate`, through `State`, `environ`, and `transform`.
- `brainpy.math.sparse`, `brainpy.math.event`, and `brainpy.math.jitconn` build
  `brainevent` structures such as `CSR`, `CSC`, and the just-in-time connectivity types,
  then hand the computation over to them.
- `brainpy.math.surrogate` is an alias of `braintools.surrogate`; `brainpy.losses` and
  `brainpy.measure` delegate to `braintools.metric`, `brainpy.initialize` to
  `braintools.init`, `brainpy.inputs` to `braintools.input`, and
  `brainpy.visualization` to `braintools.visualize`.
- The einops-style helpers in `brainpy.math` are re-exported from `brainunit.math`.

Using [`brainx`][brainx] packages internally is not the same as being compatible with the
ecosystem. These modules borrow implementations, but their models still run on
`brainpy.math.Variable`, so they do not gain unit safety, online learning, or
composability with `brainstate`-based models. Only `brainpy.state` provides those.

## Limits

Three boundaries separate the original `brainpy` API from the rest of the ecosystem:

**No `brainunit` quantities in `brainpy.math`.** `brainpy` depends on
[`brainunit`][brainunit] internally, but its array layer does not accept quantities:

```python
>>> import brainpy.math as bm, brainunit as u
>>> bm.exp(10. * u.mV)
TypeError: exp requires ndarray or scalar arguments, got <class 'saiunit.Quantity'> at position 0.
```

Strip units at the boundary — `(10. * u.mV).to_decimal(u.mV)` — or move the model to
`brainpy.state`. End-to-end unit safety exists only on the `brainstate` side.

**No online learning for `brainpy` models.** [`braintrace`][braintrace] attaches
eligibility traces to operations inside `brainstate` modules. It cannot see a
`brainpy.math.Variable` graph.

**No mixed model trees.** Modules from the two state systems cannot be composed into one
model. A `brainpy.state` module placed inside a `brainpy.DynSysGroup` never gets its
states initialized and fails at run time:

```python
AttributeError: 'LIF' object has no attribute 'V'
```

The two can still live in the same script as separate models. The same applies to
[`braincell`][braincell] and [`brainmass`][brainmass] models: usable alongside a `brainpy`
project, but not inside one `brainpy` model.

## Should you port?

Existing `brainpy` projects do not need to be rewritten; they keep working on the same
foundations as the rest of the ecosystem. Porting a model to the `brainstate`-based
packages pays off when it needs unit-safe parameters, multicompartment cells, neural-mass
infrastructure, or online learning. Port whole models, not individual layers, and keep a
reduced `brainpy` model around if you still need `brainpy.analysis`.

For a hands-on introduction to the target API, work through the
[point-neuron network tutorial](./tutorials/brainpy_point_neuron_networks.ipynb).

## Background

`brainpy` was initially developed in [Professor Si Wu's lab at Peking University][wu-lab]
and is now mainly maintained by [Chaoming Wang's lab][chaobrain] and many collaborating
laboratories.

- **2021:** The BrainPy 1.0 approach was presented at ICONIP in
  [*A Just-In-Time Compilation Approach for Neural Dynamics Simulation*][brainpy-1-paper].
- **2022–2023:** The BrainPy 2.x framework appeared as a
  [2022 preprint][brainpy-preprint] and a peer-reviewed
  [2023 eLife article][brainpy-paper].
- **2025:** The work expanded into the BrainX ecosystem. Stateful dynamical systems and
  event-driven operators moved into shared infrastructure, and the modeling scales
  `brainpy` had explored grew into focused packages: neural-mass modeling into
  [`brainmass`][brainmass], Hodgkin-Huxley cells into [`braincell`][braincell], and
  point-neuron modeling into [`brainpy.state`][brainpy.state]. The
  [change log](./CHANGELOG.md) records the first integrated BrainX releases.

[brainx]: https://brainx.chaobrain.com/
[brainpy]: https://brainpy.readthedocs.io/
[brainpy.state]: https://brainpy.readthedocs.io/state/index.html
[brainspike]: https://brainx.chaobrain.com/brainspike/
[brainstate]: https://brainx.chaobrain.com/brainstate/
[brainevent]: https://brainx.chaobrain.com/brainevent/
[braintools]: https://brainx.chaobrain.com/braintools/
[braincell]: https://brainx.chaobrain.com/braincell/
[brainmass]: https://brainx.chaobrain.com/brainmass/
[brainunit]: https://brainx.chaobrain.com/brainunit/
[braintrace]: https://brainx.chaobrain.com/braintrace/
[wu-lab]: https://www.psy.pku.edu.cn/english/people/faculty/professor/wusi/index.htm
[chaobrain]: https://chaobrain.com
[brainpy-1-paper]: https://doi.org/10.1007/978-3-030-92238-2_2
[brainpy-preprint]: https://doi.org/10.1101/2022.10.28.514024
[brainpy-paper]: https://doi.org/10.7554/eLife.86365
