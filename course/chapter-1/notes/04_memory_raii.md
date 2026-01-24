# 04 — Optional Preview: References, Scope, RAII

## Goals
- See how scope affects object lifetime.
- Understand what a reference is.

## Topics
- Scope and lifetime (when objects are created/destroyed)
- References vs values
- RAII: resources are acquired in constructors and released in destructors

## Notes
This is a **preview**. Full memory management and RAII appear later. Here we only show the idea in a small example so learners understand why “cleanup happens automatically.”

## Exercises
- Add a nested scope and observe destruction order.
- Make a variable `const` and note what changes.
