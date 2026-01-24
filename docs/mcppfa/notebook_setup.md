# notebook_setup.hpp

Notebook-only setup for xeus-cling / cling environments.

## Include
- mcppfa/notebook/notebook_setup.hpp
- Legacy: mcppfa/notebook_setup.hpp

## What it does
- Adds include paths for local headers and vendored libraries.
- Optionally wires Postgres and Matplot++ shared libraries.
- Intentionally re-include friendly for notebook cells.

## Key APIs
- Include-only header with configuration macros:
  - `MCPPFA_NOTEBOOK_ENABLE_PG`
  - `MCPPFA_NOTEBOOK_ENABLE_MATPLOT`

## Usage
```cpp
// Optional opt-outs before include
// #define MCPPFA_NOTEBOOK_ENABLE_PG 0
// #define MCPPFA_NOTEBOOK_ENABLE_MATPLOT 0
#include "include/mcppfa/notebook_setup.hpp"
```

## Notes
- Do not add `#pragma once` when using this header directly; it is meant to be re-included.
