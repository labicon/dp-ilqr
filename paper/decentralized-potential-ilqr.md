# Decentralized Potential iLQR

## Abstract

Realistic robotics navigation problems often require real-time cooperation between
agents. In previous work, we demonstrated Potential Iterative Linear
Quadratic Gaussian (iLQR) as a _Potential Minimizing Controller_ that finds a Nash
Equilibrium in the Dynamical Game. However, this was applied in a centralized framework
with one planner coordinating the paths and inputs for each of the agents. Due to this
construction, this approach's efficiency is inherently limited by the number of agents
in the navigation problem and fails to scale to real-time use cases. We introduce a
framework that allows for the _decentralization_ of the centralized problem into
smaller subproblems and validate its strucuture in a Monte Carlo simulation and on
hardware with quadcopters.

## Introduction

- Applications/Motivation
- Decentralized Control
- Nash & Potential iLQR

## Related Work

- [Potential iLQR: A Potential-Minimizing Controller for Planning Multi-Agent Interactive Trajectories](https://arxiv.org/pdf/2107.04926.pdf)
- [Decentralized receding horizon control for large scale dynamically decoupled systems](https://www.sciencedirect.com/science/article/pii/S0005109806003049)
- [Efficient Iterative Linear-Quadratic Approximations for Nonlinear Multi-Player General-Sum Differential Games](https://arxiv.org/abs/1909.04694)
- [Real-time obstacle avoidance for manipulators and mobile robots](https://ieeexplore.ieee.org/document/1087247)

## Problem Formulation

- Dynamics and Cost Definitions
- Centralized Planner
- Information Exchange/Communication

## Approach

- Interaction Graph
- Delegation into Subproblems

## Simulation

- Analysis 1: Allow unlimited solve time and stop after the solver converges or diverges.
  - Compare solve times between centralized and decentralized
  - Hopefully the distance to go and costs are relatively consistent
- Analysis 2: Cap the solve time based on a "real-time" constraint.
  - Compare distance to go and costs

## Experiments

[//]: # (Randy)

### Hardware
We test our algorithm on a Crazyflie 2.0, a micro-scale quadcopter platform. The algorithm will run on a terminal in real time while sending commands to the Crazyflie drone. The VICON system is used for real-time state estimation of the drone. 





## Conclusion

_______________________________________________________________________________________

## Brainstorming

### Relevant Papers

- iLQR Implementation - [Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf)

### Limitations

- **Sensitivity to Tuning**: The realism of the resulting trajectory is sensitive to the
  tuning of the iLQR matrices. It's unclear how to automate the selection of such $Q$
  and $R$ matrices such that the resulting trajectories are sufficiently valid.
- **Inability to guarantee collision avoidance**: while potential functions are useful
  for defining a cost surface that accounts for the costs of individual agents and
  **encouraging** cooperative interactions, it can't **guarantee** collision avoidance.
  However, this could be addressed by introducing constraints into the optimization
  framework.
- **Communication Between Agents**: In solving the decentralized problem, we utilized
  the states of all of the agents to define the interaction graph. In a more realistic
  use case, the interaction graph might be defined by the communication graph or by
  limited observibility instead.
- **Granularity of Potential iLQR**: In scaling to larger numbers of agents, the
  centralized solver begins to sacrifice some of the granularity in its resulting
  trajectories. So while decentralization helps here, this will ultimately suffer from
  the same problem given a graph of large enough diameter.
