nerva_torch documentation
=========================

A tiny, educational set of neural network components built on PyTorch.

Install and build
-----------------

.. code-block:: bash

    # from repository root
    python -m pip install -U sphinx sphinx-rtd-theme
    # build HTML docs into docs_sphinx/_build/html
    sphinx-build -b html docs_sphinx docs_sphinx/_build/html

API reference
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   nerva_torch
   nerva_torch.activation_functions
   nerva_torch.datasets
   nerva_torch.layers
   nerva_torch.learning_rate
   nerva_torch.loss_functions
   nerva_torch.loss_functions_torch
   nerva_torch.matrix_operations
   nerva_torch.multilayer_perceptron
   nerva_torch.optimizers
   nerva_torch.softmax_functions
   nerva_torch.training
   nerva_torch.utilities
   nerva_torch.weight_initializers
