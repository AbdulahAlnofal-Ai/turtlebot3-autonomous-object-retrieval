# TurtleBot3 Autonomous Object Retrieval and Placement

An autonomous robotics project developed for **AI-407 – Introduction to Robotics**, using ROS 2, Gazebo, TurtleBot3, and OpenManipulator.

## Project Overview

This project implements an autonomous robot that explores a maze, detects objects, picks them up using a robotic arm, delivers them to a target location, and returns to its starting position.

The robot operates in a simulated 7 × 7 maze and uses sensor data, search algorithms, and a finite state machine to coordinate navigation and manipulation.

## Key Features

- **Autonomous Maze Exploration:** Uses Depth-First Search (DFS) to explore unknown maze cells.
- **LiDAR-Based Detection:** Uses LiDAR readings to identify walls, open paths, and object-like obstacles.
- **Graph Construction:** Represents discovered maze cells and their connections as a graph.
- **Shortest-Path Planning:** Uses Breadth-First Search (BFS) to plan routes through discovered paths.
- **Object Manipulation:** Uses the OpenManipulator arm and gripper to pick up and deliver two objects.
- **Autonomous Return:** Returns the robot to its starting cell after completing the delivery mission.
- **Gazebo Simulation:** Tests the robot, maze, and object manipulation in a simulated environment.

## Technologies Used

- ROS 2
- Python
- Gazebo
- TurtleBot3
- OpenManipulator
- LiDAR and odometry
- MoveIt inverse kinematics
- DFS and BFS
- Finite State Machine
- Gazebo Grasp Fix plugin

## How It Works

1. Initialize the robot and wait for LiDAR and odometry data.
2. Explore the maze using DFS while building a graph of discovered cells.
3. Detect an object using LiDAR and move back to a safe approach cell.
4. Calculate arm joint positions using MoveIt inverse kinematics.
5. Pick up the first object and use BFS to plan a route to the target.
6. Drop the first object, return to the object location, and retrieve the second object.
7. Deliver the second object and navigate back to the starting cell.

## Example Scenario

The project report describes a test scenario with:

- **Starting cell:** `(7, 4)`
- **Object cell:** `(5, 7)`
- **Target cell:** `(6, 4)`
- **Object approach cell:** `(6, 7)`
- **Target approach cell:** `(6, 5)`

The launch configuration supports changing the start, target, object locations, and robot orientation.

## ROS 2 Communication

| Interface | Purpose |
|---|---|
| `/scan` | Receives LiDAR readings |
| `/odom` | Receives robot position and orientation |
| `/cmd_vel` | Controls robot movement |
| `/arm_controller/joint_trajectory` | Commands arm joint trajectories |
| `/gripper_controller/gripper_cmd` | Opens and closes the gripper |
| `/compute_ik` | Requests inverse kinematics from MoveIt |

## Project Structure

See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for the planned repository layout.

## Demo

The project demonstration will be linked here when available.

## Team

- Abdulah Al Nofal
- Hassan Mohammed Nasr
- Yousef Abu Ghali

**Course:** AI-407 – Introduction to Robotics

## Acknowledgments

Developed as an academic robotics project using ROS 2, Gazebo, TurtleBot3, and OpenManipulator. 


## Setup and Installation

### Prerequisites

Before running the project, prepare an environment with the required ROS 2 distribution and compatible versions of Gazebo, TurtleBot3, and OpenManipulator packages.

You will also need the project dependencies and robot description files referenced by the launch configuration.

### Clone the Repository

```bash
git clone https://github.com/AbdulahAlnofal-Ai/turtlebot3-autonomous-object-retrieval.git
cd turtlebot3-autonomous-object-retrieval
```

### Build the ROS 2 Package

From the repository root, copy or place the package in a ROS 2 workspace under `src/`, then build it:

```bash
cd ~/ai407_ws
colcon build --packages-select pkg
source install/setup.bash
```

**Note:** The repository currently stores the package files at its root. The build commands above assume you have placed the package inside `~/ai407_ws/src/pkg/` first. Additional robot description files and dependencies may be required.

## Demo

A demonstration video can be linked here once it is available.

## Limitations

The simulation depends on compatible ROS 2, Gazebo, TurtleBot3, and OpenManipulator configurations. Object detection and manipulation performance may vary depending on sensor readings, robot positioning, and grasp stability.
