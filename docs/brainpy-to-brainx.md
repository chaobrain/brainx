# BrainPy to BrainX

[`brainpy`][brainpy], initially developed in [Professor Si Wu's lab at Peking University][wu-lab], is the experimental precursor of [`brainx`][brainx], now mainly maintained by [Chaoming Wang's lab][chaobrain] and many collaborating laboratories. 

The ideas first explored in it — stateful dynamical systems, and event-driven operators — were developed into the focused, production-level packages
that make up [`brainx`][brainx] today.

The neural-mass modeling explored in [`brainpy`][brainpy] evolved into the dedicated
[`brainmass`][brainmass] package. Its Hodgkin-Huxley cell models inspired the more
comprehensive conductance-based models, ion-channel systems, and neuronal morphology
support now provided by [`braincell`][braincell]. Its point-neuron modeling — the part
[`brainpy`][brainpy] was best at — became [`brainpy.state`][brainpy.state].

The development unfolded in three broad stages:

- **2021:** The BrainPy 1.0 approach was presented at ICONIP in
  [*A Just-In-Time Compilation Approach for Neural Dynamics Simulation*][brainpy-1-paper].
- **2022–2023:** The broader BrainPy 2.x framework appeared as a
  [2022 preprint][brainpy-preprint] and a peer-reviewed
  [2023 eLife article][brainpy-paper].
- **2025:** The work expanded into the BrainX ecosystem, separating shared
  infrastructure and different modeling scales into focused packages; the
  [change log](./CHANGELOG.md) records the first integrated BrainX releases.

This page explains how the packages relate and whether your project should move.
When you decide to move, the [migration notes](./brainpy-migration-notes.md) cover the
mechanics: API mappings, a worked before/after example, and the cases that need care.

## `brainpy` and `brainpy.state`

`brainpy` and `brainpy.state` are two different packages. They share an import root
and little else: they are distributed separately on PyPI (`brainpy` and
`brainpy-state`), and they are built on different state systems — `brainpy` on its own
`brainpy.math.Variable`, `brainpy.state` on [`brainstate`][brainstate].

Most of the confusion around migration comes from this one fact. Both are installed by
`pip install -U BrainX` and import cleanly into the same process, so a project can
migrate one model at a time. The
[migration notes](./brainpy-migration-notes.md) tabulate the differences.

## Status and support

This page describes `brainpy` 2.x and the component versions pinned by the current
[`brainx`][brainx] release; see the [change log](./CHANGELOG.md) for the exact pins.

`brainpy` ships in the BrainX pin set and is in **maintenance**: it receives
compatibility and bug fixes (for example the recent JAX 0.11 fix), and its `analysis`
module is still unique in the ecosystem. New modeling features land in the
[`brainx`][brainx] packages, not in `brainpy`.

Existing `brainpy` projects do not need to be rewritten. Migration pays off when a
project needs unit-safe parameters, multicompartment cells, online learning, or
long-term feature work.

## Where each modeling scale now lives

**Point-neuron networks → [`brainpy.state`][brainpy.state].** This was `brainpy`'s main
strength, and it is the scale that carried over most directly. Model names and
projection patterns stay recognizable; the state system and the runner are what change.

**Cells, ions, and morphology → [`braincell`][braincell].** Migrate rather than mix. The
older ion and channel APIs in `brainpy` have known design limitations and its
compartmental support is restricted to single-compartment models.
[`braincell`][braincell] provides comprehensive conductance-based and Hodgkin-Huxley
models together with morphologically structured, multicompartment cells.

**Neural-mass and whole-brain models → [`brainmass`][brainmass].** The `brainpy` rate
models cover FitzHugh-Nagumo, Stuart-Landau, threshold-linear, and Wilson-Cowan
dynamics. [`brainmass`][brainmass] provides direct counterparts, plus models `brainpy`
never had — Jansen-Rit, Epileptor, Hopf, Montbrió-Pazó-Roxin, Wong-Wang,
Larter-Breakspear — and the infrastructure whole-brain work needs: explicit coupling
schemes, structured noise processes, forward models for BOLD, EEG, and MEG signals, and
parameter fitting against empirical data. Unlike the cellular case this is not a
warning; the `brainpy` rate models still work, this scale is simply developed in
[`brainmass`][brainmass] now.

## Why existing brainpy code still works

The current `brainpy` codebase was reconstructed on top of [`brainstate`][brainstate],
[`brainevent`][brainevent], and [`braintools`][braintools]. This is not only a dependency
relationship: many internal functions delegate to the production-level implementations,
so the two share the same underlying code rather than maintaining parallel ports.

- `brainpy.math.surrogate` is an alias of `braintools.surrogate`, and the einops-style
  helpers in `brainpy.math` are re-exported from `brainunit.math`.
- The operators in `brainpy.math.sparse`, `brainpy.math.event`, and
  `brainpy.math.jitconn` build [`brainevent`][brainevent] structures such as `CSR`,
  `CSC`, and the just-in-time connectivity types, then hand the computation over to them.
- `brainpy.losses` and `brainpy.measure` delegate to `braintools.metric`, and
  `brainpy.initialize` delegates to `braintools.init`.
- `brainpy.inputs` builds its current waveforms from `braintools.input`, and
  `brainpy.visualization` from `braintools.visualize`.
- State handling, environment settings, and compiled transformations come from
  [`brainstate`][brainstate], through `State`, `environ`, and `transform`.

So a `brainpy` project keeps working and stays on the same foundations as the rest of
the ecosystem. Two packages sit outside that arrangement: [`brainunit`][brainunit],
whose quantities `brainpy.math` cannot consume, and [`braintrace`][braintrace], whose
online-learning traces attach to [`brainstate`][brainstate] modules only. Sharing
foundations also does not mean models from both sides can be composed into one — see
the [migration notes](./brainpy-migration-notes.md) for both limits in detail.

## What brainpy still owns

`brainpy.analysis` — phase-plane analyzers, `Bifurcation1D` / `Bifurcation2D`, slow-fast
decomposition — has no counterpart in [`brainx`][brainx]. This is the one capability for
which `brainpy` remains the right tool, and the reason it stays in the pin set rather
than being retired.

## Should I migrate?

| your situation | recommendation |
| --- | --- |
| Starting a new point-neuron project | [`brainpy.state`][brainpy.state] |
| Maintaining a working `brainpy` project | stay; migrate when you need something below |
| Modeling ions, ion channels, or morphology | [`brainpy.state`][brainpy.state] + [`braincell`][braincell] — migrate |
| Modeling neural-mass or whole-brain dynamics | [`brainmass`][brainmass] |
| Wanting unit-safe parameters ([`brainunit`][brainunit]) | migrate — `brainpy.math` cannot consume quantities |
| Wanting online learning ([`braintrace`][braintrace]) | migrate — traces attach to `brainstate` modules only |
| Depending on `brainpy.analysis` | stay on `brainpy` for that part |

## In short

Treat `brainpy` as the experimental embryo that inspired the [`brainx`][brainx]
ecosystem. It remains maintained and usable for established projects and for its unique
analysis tools, while [`brainx`][brainx] is where new work belongs:
[`brainpy.state`][brainpy.state] for point-neuron networks, [`braincell`][braincell] for
ions, channels, and morphology, and [`brainmass`][brainmass] for neural-mass and
whole-brain models.

Ready to move? Continue with the
[BrainPy migration notes](./brainpy-migration-notes.md).

[brainx]: https://brainx.chaobrain.com/
[brainpy]: https://brainpy.readthedocs.io/
[brainpy.state]: https://brainx.chaobrain.com/brainpy-state/
[brainstate]: https://brainx.chaobrain.com/brainstate/
[brainevent]: https://brainx.chaobrain.com/brainevent/
[braintools]: https://brainx.chaobrain.com/braintools/
[braincell]: https://brainx.chaobrain.com/braincell/
[brainmass]: https://brainx.chaobrain.com/brainmass/
[brainunit]: https://brainx.chaobrain.com/brainunit/
[braintrace]: https://brainx.chaobrain.com/braintrace/
[wu-lab]: https://www.psy.pku.edu.cn/english/people/faculty/professor/wusi/index.htm
[wangchaoming]: https://wangchaoming.com
[chaobrain]: https://chaobrain.com
[brainpy-1-paper]: https://doi.org/10.1007/978-3-030-92238-2_2
[brainpy-preprint]: https://doi.org/10.1101/2022.10.28.514024
[brainpy-paper]: https://doi.org/10.7554/eLife.86365
