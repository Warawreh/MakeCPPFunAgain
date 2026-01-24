# 01 — Tooling & xcpp workflow

## Goals
- Understand how xcpp/xeus-cling executes C++ in notebooks.
- Learn the “edit → run → observe” learning loop.
- Know how include paths work in this repo.

## How notebooks work
Jupyter notebooks mix **markdown explanations** with **runnable C++ cells**. With xcpp, each cell is compiled and executed immediately. This gives instant feedback and lets learners change code and re-run without building a whole project.

**Workflow:**
1. Read the explanation
2. Run the example cell
3. Edit the exercise cell
4. Re-run and observe the output

## Include paths in this repo
Project headers live under `include/`. In notebooks, you can add that path once or include only the headers you need. For clean learning cells, prefer including just the headers you use.

## Exercises
- Run a cell that prints a message.
- Change the message and re-run.
- Include a small helper header and call one function.
