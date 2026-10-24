Brain Simulation Ecosystem
==========================

BrainX is an open ecosystem for building, running, and studying brain
simulations across scales. It brings together JAX-native libraries for
unit-safe computation, sparse event processing, reusable experiment tooling,
stateful model construction, biophysical and neural-mass modeling, and
learning-driven dynamics. For the project landing page and broader ecosystem
overview, see `brainx.chaobrain.com <https://brainx.chaobrain.com/>`_.

**This page is a summary of the BrainX documentation**. It introduces the core
packages, shows how the layers fit together, and points you to tutorials that
move from foundational tools to complete simulation workflows.

If you are setting up BrainX for the first time, start with the
:doc:`installation tutorial <install>` to prepare your environment, install the
required packages.

The tutorials are organized around the BrainX package layers:

 - Physical Units :  ``brainunit``
 - Events & Tools :  ``brainevent`` ``braintools``
 - State Runtime :  ``brainstate``
 - Models :  ``brainpy`` ``brainspike`` ``braincell`` ``brainmass``
 - Online Learning :  ``braintrace``

**For more detailed information**, click a package card to open its full package documentation:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: brainunit
      :link: https://brainx.chaobrain.com/brainunit/
      :link-type: url
      :class-card: sd-card-hover sd-shadow-md

      Unit-aware numerical computing for values, parameters, and equations.


   .. grid-item-card:: brainevent
      :link: https://brainx.chaobrain.com/brainevent/
      :link-type: url
      :class-card: sd-card-hover sd-shadow-md

      Event-driven computation for sparse spike-based workloads.


   .. grid-item-card:: braintools
      :link: https://brainx.chaobrain.com/braintools/
      :link-type: url
      :class-card: sd-card-hover sd-shadow-md

      Experiment utilities for inputs, tasks, connectivity, metrics, and plots.


   .. grid-item-card:: brainstate
      :link: https://brainx.chaobrain.com/brainstate/
      :link-type: url
      :class-card: sd-card-hover sd-shadow-md

      Stateful JAX modules, parameters, transforms, and runtime management.


   .. grid-item-card:: brainpy
      :link: https://brainpy.readthedocs.io/
      :link-type: url
      :class-card: sd-card-hover sd-shadow-md

      General-purpose brain dynamics programming, with state-based models in ``brainpy.state``.


   .. grid-item-card:: brainspike
      :link: https://brainx.chaobrain.com/brainspike/
      :link-type: url
      :class-card: sd-card-hover sd-shadow-md

      Point-neuron spiking networks: NEST-compatible models, simulation, and training.


   .. grid-item-card:: braincell
      :link: https://brainx.chaobrain.com/braincell/
      :link-type: url
      :class-card: sd-card-hover sd-shadow-md

      Biophysical cell modeling with ion channels and morphology support.


   .. grid-item-card:: brainmass
      :link: https://brainx.chaobrain.com/brainmass/
      :link-type: url
      :class-card: sd-card-hover sd-shadow-md

      Neural mass models and whole-brain dynamics.


   .. grid-item-card:: braintrace
      :link: https://brainx.chaobrain.com/braintrace/
      :link-type: url
      :class-card: sd-card-hover sd-shadow-md

      Online learning and eligibility traces for RNNs and SNNs.


.. toctree::
   :hidden:
   :caption: Getting started
   :maxdepth: 1

   install.md
   brainpy-to-brainx.md
   brainpy-migration-notes.md
   CHANGELOG.md


.. toctree::
   :hidden:
   :caption: Tutorials
   :maxdepth: 1

   tutorials/brainunit_unit_aware_computations.ipynb
   tutorials/brainevent_event-driven.ipynb
   tutorials/braintools_expriment_utilities.ipynb
   tutorials/brainstate_transformations.ipynb
   tutorials/brainpy_point_neuron_networks.ipynb
   tutorials/brainspike_NEST_compatible.ipynb
   tutorials/braincell_HH_neuron.ipynb
   tutorials/braincell_morphological_golgi_cell.ipynb
   tutorials/brainmass_jansenrit_node_simulation.ipynb
   tutorials/brainmass_Modeling_MEG_data.ipynb
   tutorials/braintrace_online_learning.ipynb


.. toctree::
   :hidden:
   :caption: Developer resources
   :maxdepth: 1

   contributing.md
