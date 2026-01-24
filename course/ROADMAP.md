# C++ Course Roadmap (xcpp17 & Jupyter)

## Introduction: Tools & Interactive Learning

### Environment & Setup
Present the C++17 kernel (xcpp17 via xeus-cling) inside Jupyter. Explain that notebooks combine **narrative text + live code**, so learners can write C++ and run it immediately without manual compile steps. Emphasize the instant feedback loop (“write code → Shift+Enter”).

### Benefits of Interactive Notebooks
Notebooks embed explanations alongside runnable C++ examples. Learners can edit code in place and see results instantly. xeus-cling also provides quick help (cppreference) and rich output in cells.

### Learning Approach
Each chapter uses this structure:
1) Short concept explanation (Markdown)
2) Runnable code example (C++ cell)
3) Practice exercise cell
4) Short quiz (MCQ)

Projects/mini-apps cap each major stage to integrate skills.

---

## Stage 1: Basics of C++ (Hello World, I/O, Variables, Types)

### Topics
- Syntax, hello world, comments
- Basic I/O using `std::cout` / `std::cin`
- Variables and fundamental types (`int`, `double`, `bool`, `std::string`)
- Brief C++17 features like `auto`

### Examples
- Printing text and expressions
- Reading input and doing simple arithmetic
- Formatting output

### Exercises & Quiz
- Modify a greeting to use the user’s name
- Compute area of a circle
- MCQ: predict outputs, identify type issues

### Project
- A simple calculator or greeting program

---

## Stage 2: Control Flow (Conditionals and Loops)

### Topics
- `if / else if / else`, `switch`
- `for`, `while`, `do-while`
- Boolean logic and common loop patterns

### Examples & Visuals
- Age check program
- Summation loops
- Simple progress prints or ASCII tables to visualize loop steps

### Exercises & Quiz
- Print even numbers up to `N`
- Compute factorial
- MCQ: predict loop outputs and pick correct loop type

### Project
- Number guessing game or menu-driven program

---

## Stage 3: Functions, Arrays/Vectors, and Modular Code

### Topics
- Defining and calling functions
- Scope and lifetime
- Arrays vs `std::vector`
- Basic string handling and line input

### Examples
- Average of a vector
- Passing vectors into functions

### Exercises & Quiz
- Prime checker
- Fibonacci array
- MCQ: match calls to signatures, indexing questions

### Project
- Simple gradebook/report generator

---

## Stage 4: Basic Data Structures & Algorithms (Sorting & Searching)

### Topics
- Vectors vs arrays
- Sorting: bubble, selection, insertion
- Searching: linear, binary
- Time complexity intuition

### Examples & Visuals
- Step-by-step sort states
- Trace a binary search

### Exercises & Quiz
- Implement bubble sort and print steps
- Write binary search and test
- MCQ: comparisons and trace results

### Project
- Contact list or inventory manager with sort/search

---

## Final Projects & Challenges

### Capstones
- Text-based game (Hangman, Tic-Tac-Toe)
- Data challenge: filter/sort/summarize a dataset
- Algorithm puzzle: optimize or visualize a sort

### Review & Next Steps
Summarize the journey from syntax to algorithms. Encourage exploration of files, `std::map`, and advanced topics.

---

## Learning-by-Doing Standard
Every notebook mixes explanation, runnable C++ code, an exercise cell, and a short quiz. This creates a consistent interactive learning loop for students.
