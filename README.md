# A* Online Path Planning in Dynamic Environment

This project implements an online path planning system for a robot navigating in a dynamic grid-world environment. The planner considers both geometric distance and environmental uncertainty, allowing the robot to avoid high-risk regions and replan when dynamic obstacles appear.

## Overview

Traditional A* path planning mainly optimizes path length based on a known static map. However, in realistic indoor environments, obstacles may move, and some regions may be more uncertain or risky than others.

This project extends the standard A* framework by introducing a risk-aware belief map. The planner combines local observations with an experiential risk map to generate safer and more adaptive paths in dynamic environments.

## Key Features

- Grid-world simulation environment
- Static and dynamic obstacle modeling
- Online A* path planning
- Risk-aware belief map fusion
- Experiential map for high-risk region prediction
- Path smoothing using spline-based methods
- PID-based trajectory tracking
- Visualization of planned paths, obstacles, and belief maps

## Method

The system consists of four main modules:

1. **Environment Modeling**  
   The environment is represented as a 2D occupancy grid. Static obstacles and dynamic obstacles are generated in the map.

2. **Belief Map Construction**  
   A belief map is used to represent the probability or risk of obstacle occurrence in each grid cell.

3. **Risk-Aware A* Planning**  
   The A* planner uses a cost function that combines path distance, obstacle risk, and belief-map uncertainty.

4. **Online Replanning**  
   When dynamic obstacles change the environment, the planner updates the belief map and replans the path accordingly.

## Installation

Create a conda environment:

```bash
conda create -n dynamic-Astar python=3.10
conda activate dynamic-Astar
pip install numpy scipy matplotlib
```

## Usage

Run the simulation:
```bash
python main_online.py
```

## Results
The risk-aware planner produces paths that avoid high-risk and uncertain regions. Compared with standard A*, the proposed method may generate longer but safer paths in environments with dynamic or uncertain obstacles.

## Future Work
- Improve dynamic obstacle prediction
- Integrate D* Lite for more efficient replanning
- Add more realistic robot dynamics
- Extend the system to MuJoCo or Isaac simulation
- Compare with other planning algorithms such as D* Lite, RRT*, and MPC-based planning