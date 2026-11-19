# Reproduction and provenance

## Runtime files

`setup_quadruped.m` adds only `simulation/` and `src/` to the path, executes the supplied parameter script in the base workspace, loads `agents/trained_agent.mat`, binds `agent1_Trained` to the RL Agent block and constructs the Simulink reinforcement-learning environment using the saved policy's observation/action specifications.

The supplied policy is an `rl.agent.rlTD3Agent` with 44 observations and 8 actions. The main Simulink model comes from the archive's `try_2/` folder. Its geometry uses the supplied extrusion and inverse-kinematics helpers, so those files are included together.

The simulation may terminate early according to the original model's guards. Loading and executing a policy does not, by itself, demonstrate robust walking, hardware deployment or reproduce a thesis benchmark.

## Changes made for launchability

- Added root setup, evaluation and training entry points.
- Converted `quadrupedResetFcn.mlx` into a callable `.m` function without changing its reset logic.
- Removed the two script assignments preceding the function definition in the historical `createQuadNetworks.m`, so it is a callable function file.
- Kept the supplied model, parameter/geometry helpers and exported trained policy unchanged.
- Left historical scripts and existing repository content available; the new entry points avoid their missing workspace variables and placeholder plotting blocks.

## Saved experiment artifacts

The original archive contains three byte-identical copies of `ReinforcementLearningDesignerSession.mat`; one is sufficient. The distinct DDPG-labelled session is retained separately. They remain in the original local archive and are omitted from the GitHub upload; they are not required by the demo.

Original training screenshots show intermediate runs. Saved `.fig` files and network diagrams are preserved, but their checkpoint provenance is not independently established. The old `quadruped_robot_TD3_first_trial.m` contains randomly generated placeholder speed, energy and attitude arrays. Corresponding illustrations must not be presented as measured results without locating their true logged data.

## MATLAB validation

The exported MAT-file was inspected and identified as a TD3 agent with a 44-by-1 observation specification and an 8-by-1 action specification. Simulation validation was stopped at the project owner's request. No successful model-update, simulation, or fresh-training test is claimed. The launch adapters are supplied unvalidated; MATLAB/toolbox compatibility and any runtime issues must be checked when the project is run.
