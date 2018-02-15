Outline:

* Abstract
* Introduction
    * Here's what minimization does...
    * Point to users
* Problem
    * No standard interface for optimizers or functions
    * scipy.optimize.minimize is a black box
    * Mixing of function arguments with optimization arguments (and many arguments)
* Solution
    * Classes (briefly list attributes, functions)
* Solution enhancements
    * Provide standard interface
        * for enhancements to sklearn, dask-ml, etc. Possibly PyTorch.
    * Clean up minimize API (it's complicated rn)
    * Provide class features
        * expert interaction
        * expose alg hyperparameters (grid search, etc)
        * keyboard interrupts
* Solution implementation
    * List functions, attributes in depth
    * Existing code
    * Speed
    * Backwards compatibility
* Existing work
    * PyTorch
    * scikit-optimize (they refer to scipy.optimize.minimize)
    * sklearn
    * dask-ml
