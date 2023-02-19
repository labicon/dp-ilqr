# *Distributed Potential-iterative Linear Quadratic Regulator (DP-iLQR)*

 [![License:
 MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
Repository containing code for a distributed implementation of [Potential
iLQR](https://arxiv.org/pdf/2107.04926.pdf). Putting Potential iLQR into a nutshell, the
ultimate goal is enable cooperative real-time multi-agent navigation by posing the
coupled optimal control problem for each of the agents as a combined decoupled problem
encapsulating the interactions between all agents simultaneously using the concept of a
[potential game](https://en.wikipedia.org/wiki/Potential_game). DP-iLQR is an extension
of this algorithm that improves scalability by splitting up the centralized problem with
all agents into smaller problems with subsets of agents based on their relative
proximities.

The setup that we're most interested in is one in which multiple agents would like to
navigate around each other in a shared space. Each agent starts at some position and
would like to arrive at some goal position or state. Several applications in mobile
robotics include:
- warehouse navigation
- robot/human crowd navigation
- space robotics

![5 Unicycles Agents](media/5_unicycles.gif)

The above is one example of what this looks like using 5 unicycle models. Note that
while the dynamics of this scenario are homogeneous, this library currently supports
simulation of non-homogeneous models via zero-padding the states

## Installation

1. After cloning the repo, one must first install all necessary dependencies (ideally
   into an environment). Then, one can run the following with the environment activated:

        pip install -e .

   This will install an editable version of the package into the local environment. To
   ensure it worked, navigate to any directory and run:

	    python -c "import dpilqr; print(dpilqr.util.repopath)"

   This should print out the top level of the repository.
2. Additionally, you must compile the C++ extensions using Cython by running:
	
	    python setup.py build_ext --inplace

   This should create a `bbdynamicswrap.cpp` as well as a `*.so` file that `dpilqr` will
   automatically include in the package namespace.

## Organization
- [dpilqr](dpilqr) contains the project source code
- [scripts/examples.py](run/examples.py) provides several examples that exercise many
  aspects of the library.
- [scripts/analysis.py](run/analysis.py) contains scripts that run monte-carlo simulations
  to compare Potential-iLQR from DP-iLQR across several parameters.

## Features
* Simulation and integration of state space models by defining a ordinary differential
  equation and a linearization method in C++. These are the currently implemented models:
  * `DoubleIntDynamics4D`
  * `CarDynamics3D`
  * `UnicycleDynamics4D`
  * `BikeDynamics5D`
  * `HumanDynamics6D`
  * `QuadcopterDynamics6D`
  * `QuadcopterDynamics12D`
* Construction of various cost models including:
  * `ReferenceCost`
    * Penalizes deviations from some reference trajectory
    * $C(x, u) = (x - \bar{x})^\intercal Q (x - \bar{x}) + u^\intercal R u$
  * `ProximityCost`
    * Penalizes the distances $d^{ij}$ between agents $i$ and $j$ below some threshold $d_{\text{prox}}$
    * $C(d^{ij}) = \begin{cases} \beta( d^{ij} - d_{\text{prox}})^2 & d^{ij} < d_{\text{prox}} \\ 0 & \text{otherwise} \end{cases}$
  * `Cost`
    * Any other cost implementing the appropriate methods for the particular problem.
* Potential iLQR solver based on [this
  paper](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf) by Yuval Tassa
  that incorporates the above dynamical models and cost structures
* DP-iLQR solver that takes advantage of the sparsity of the state space to solve subproblems
  individually. The GIF below visualizes what that might look like for 'agents' in Brownian motion
  where the red is the proximity cost and the grey is the neighborhood that creates the subproblem.
  We see as the agents move around, their interaction graphs are dynamically updated.
  ![Interaction Graph Example](media/graphs-5-agents.gif)


## Credit
The following two repositories have been instrumental from both an algorithms and
software architecture perspective in the development of this project:
- [HJReachability/ilqgames](https://github.com/HJReachability/ilqgames)
- [anassinator/ilqr](https://github.com/anassinator/ilqr)

