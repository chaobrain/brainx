# Brain Simulation Ecosystem (BrainX)

[![PyPI version](https://img.shields.io/pypi/v/brainx)](https://pypi.org/project/brainx/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/chaobrain/brainx/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-brainx.chaobrain.com-blue)](https://brainx.chaobrain.com)
[![Continuous Integration](https://github.com/chaobrain/brainx/actions/workflows/CI.yml/badge.svg)](https://github.com/chaobrain/brainx/actions/workflows/CI.yml)


<p align="center">
  	<img alt="Header image of Brain Modeling Ecosystem." src="https://brainx.chaobrain.com/images/brainx-ecosystem.webp" width=50%>
</p> 

## Overview

The BrainX ecosystem provides a comprehensive framework for brain simulation and modeling.
It provides tools and libraries for researchers to model, simulate, train, and analyze neural systems at different
scales.

**Core components** in this ecosystem includes:

- [BrainPy](https://github.com/brainpy/BrainPy): General-purpose brain dynamics programming, including the state-based
  models of `brainpy.state`, comes from Prof. Si Wu's lab at Peking University.

- [BrainSpike](https://github.com/chaobrain/brainspike): Point-neuron spiking neural networks: NEST-compatible models,
  simulation, and training.

- [BrainUnit](https://github.com/chaobrain/brainunit): Comprehensive physical units and unit-aware mathematical system
  for brain dynamics.

- [BrainCell](https://github.com/chaobrain/braincell): Intuitive, parallel, and efficient simulation for biologically
  detailed brain cell modeling. Collaborated with Prof. Songting Li's lab at Shanghai Jiao Tong University.

- [BrainMass](https://github.com/chaobrain/brainmass): Whole-brain modeling with differentiable neural mass models.

- [BrainState](https://github.com/chaobrain/brainstate): State-based IR compilation for efficient simulation of brain
  models on CPUs, GPUs, and TPUs.

- [BrainEvent](https://github.com/chaobrain/brainevent): Enabling event-driven computations in brain dynamics.

- [BrainTrace](https://github.com/chaobrain/braintrace): Eligibility trace-based online learning for brain dynamics:
  $O(N)$ complexity for SNNs and $O(N^2)$ for RNN computations.

- [BrainTools](https://github.com/chaobrain/braintools): Commonly used tools for brain dynamics programming, for example
  checkpointing.

- [PINNx](https://github.com/chaobrain/pinnx): Physics-informed neural networks for scientific machine learning in JAX.

- More components may be added in the future.

## Installation

The ecosystem can be installed with the following command:

```bash
pip install BrainX -U
```

This command installs the core package and pins specific versions of the component projects known to work together,
ensuring compatibility based on integration tests.

On CPU platforms, the following command can be used to install the ecosystem with all components:

```bash
pip install BrainX[cpu] -U
```

On GPU platforms, the following command can be used to install the ecosystem with all components:

```bash
pip install BrainX[cuda12] -U

pip install BrainX[cuda13] -U
```

On TPU platforms, the following command can be used to install the ecosystem with all components:

```bash
pip install BrainX[tpu] -U
```

For development, you might want to clone the repository and install it in editable mode:

```bash
git clone https://github.com/chaobrain/brainx.git
cd brainx
pip install -e .
```

### BrainX Skill

Install the BrainX skill for your coding agent with:

```bash
npx brainx-skill install
```

## Documentation

For detailed documentation, tutorials, and examples, visit
our [Documentation Portal](https://brainx.chaobrain.com).

## Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more
information on how to get involved.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you use the BrainX Ecosystem in your research, please cite it appropriately. Refer to
the [citation guide](https://brainx.chaobrain.com/research/papers-about-brainx/) on our documentation portal.

## Support

If you have questions, encounter issues, or need support, please:

* Check the [documentation](https://brainx.chaobrain.com).
* Search the [existing issues](https://github.com/chaobrain/brainx/issues).
* [Open a new issue](https://github.com/chaobrain/brainx/issues/new/choose) if your problem is not
  addressed.
* Contact us via email: `chao.brain@qq.com`.


