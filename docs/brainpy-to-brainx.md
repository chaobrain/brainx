# BrainPy in BrainX

[`brainpy`][brainpy] is one package in [`brainx`][brainx]. This page explains where it
sits among the other packages, which package fits which kind of model, and what an
existing `brainpy` project can and cannot combine with the rest of the ecosystem.

It describes `brainpy` 2.x and the component versions pinned by the current
[`brainx`][brainx] release; see the [change log](./CHANGELOG.md) for the exact pins.

## Background

`brainpy` was initially developed in [Professor Si Wu's lab at Peking University][wu-lab].
It predates BrainX and served as the testbed for the ideas the ecosystem grew from.
`brainpy` and the rest of BrainX are now mainly maintained by
[Chaoming Wang's lab][chaobrain] and the community.

- **2021:** The BrainPy 1.0 approach was presented at ICONIP in
  [*A Just-In-Time Compilation Approach for Neural Dynamics Simulation*][brainpy-1-paper].
- **2022–2023:** The BrainPy 2.x framework appeared as a
  [2022 preprint][brainpy-preprint] and a peer-reviewed
  [2023 eLife article][brainpy-paper].
- **2025:** The work expanded into the BrainX ecosystem.

## Shared foundations

The current `brainpy` codebase is built on [`brainstate`][brainstate],
[`brainevent`][brainevent], [`braintools`][braintools], and [`brainunit`][brainunit].
Many internal functions delegate to their implementations, so `brainpy` and these
packages share the same code rather than maintaining parallel ports:

- State handling, environment settings, and compiled transformations come from
  `brainstate`, through `State`, `environ`, and `transform`.
- `brainpy.math.sparse`, `brainpy.math.event`, and `brainpy.math.jitconn` build
  `brainevent` structures such as `CSR` and the just-in-time connectivity types,
  then hand the computation over to them.
- `brainpy.math.surrogate` is an alias of `braintools.surrogate`; `brainpy.losses` and
  `brainpy.measure` delegate to `braintools.metric`, `brainpy.initialize` to
  `braintools.init`, `brainpy.inputs` to `braintools.input`, and
  `brainpy.visualization` to `braintools.visualize`.

Using [`brainx`][brainx] packages internally is not the same as being compatible with the
ecosystem. These modules borrow implementations, but their models still run on
`brainpy.math.Variable`, so they do not gain unit safety, online learning, or
composability with `brainstate`-based models. Only
[`brainpy.state`](#brainpy-state-for-ecosystem-compatibility) provides those.

## Limitations

Three boundaries separate the original `brainpy` API from the rest of the BrainX ecosystem:

**1. No `brainunit` quantities in `brainpy.math`.** 

`brainpy` depends on
[`brainunit`][brainunit] internally, but its array layer does not accept quantities:

```python
>>> import brainpy.math as bm, brainunit as u
>>> bm.exp(10. * u.mV)
TypeError: exp requires ndarray or scalar arguments, got <class 'saiunit.Quantity'> at position 0.
```

Strip units at the boundary — `(10. * u.mV).to_decimal(u.mV)` — or move the model to
`brainpy.state`. End-to-end unit safety exists only on the `brainstate` side.

**2. No online learning for `brainpy` models.** 

[`braintrace`][braintrace] attaches
eligibility traces to operations inside `brainstate` modules. It cannot see a
`brainpy.math.Variable` graph.

**3. No mixed model trees.** 

Modules built on `brainpy.math.Variable` and on
`brainstate.State` cannot be composed into one model. A `brainpy.state` module placed
inside a `brainpy.DynSysGroup` never gets its states initialized and fails at run time:

```python
AttributeError: 'LIF' object has no attribute 'V'
```

The two can still live in the same script as separate models. The same applies to
[`braincell`][braincell] and [`brainmass`][brainmass] models: usable alongside a `brainpy`
project, but not inside one `brainpy` model.

(should-you-port)=
## Should you port?

Your existing `brainpy` projects do not need to be rewritten; they keep working on the same
foundations as the rest of the ecosystem. Porting a model to `brainpy.state` or another
`brainstate`-based package pays off when it needs unit-safe parameters, multicompartment
cells, neural-mass infrastructure, or online learning. Port whole models, not individual
layers, and keep a reduced `brainpy` model around if you still need `brainpy.analysis`.

(brainpy-state-for-ecosystem-compatibility)=
## `brainpy.state` for ecosystem compatibility

Since BrainPy 2.7.0 the `brainpy` distribution ships two APIs built on different state
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
process, so a project can move one model at a time. For a hands-on introduction, work
through the [point-neuron network tutorial](./tutorials/brainpy_point_neuron_networks.ipynb).

## Choosing a package in BrainX

| your situation | recommendation |
| --- | --- |
| Maintaining a working `brainpy` project | keep it; see [Should you port?](#should-you-port) above |
| Modeling large-scale point-neuron spiking neural networks | [`brainspike`][brainspike] |
| Modeling ions, ion channels, or morphology | [`braincell`][braincell] |
| Modeling neural-mass or whole-brain dynamics | [`brainmass`][brainmass] |
| Phase-plane or bifurcation analysis | `brainpy.analysis` |

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
