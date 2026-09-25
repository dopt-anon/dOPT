# Vendored FFOLayer source

This directory contains the minimal FFOLayer 0.1.2 source used by the platoon
experiment, based on upstream commit `28905f3e1750fca5b8918954d5d2ea5bed0cbacc`:

https://github.com/GT-KOALA/FFOLayer

The retained equality-constrained backward path evaluates the new equality
dual term at the perturbed primal variables. The experiment imports this copy
explicitly so that its behavior does not depend on an installed wheel. See
`LICENSE` for the upstream MIT license.
