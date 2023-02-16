# CS182-282A-Final-Project

This is the final project done by Ann Katrine, Chengyuan Li, Jesper Hauch and Trent Fellbootman (Yiming Gong) at UC Berkeley. The purpose of this project is to serve as
an assignment to be done by students taking the course "Designing, Visualizing and Understanding Deep Neural Networks". After completing this assignment, students
should have a solid understanding of how GAN and CycleGAN works, as well as their training dynamics.

This repository also contains a Tensorflow-like abstraction over JAX and Flax.

Code structure:

- The library module is where most of the code reside. It contains the JAX abstraction (`base.py` in `models` package), GAN and CycleGAN architectures and
utilities for dataset generation and visualizations. Most filenames are self-explanatory.
- The Jupyter notebooks prefixed with "homework" are what a student should be working on when doing this assignment.
