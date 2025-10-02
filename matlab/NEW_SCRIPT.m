% Fix Random Seed Generator to Improve Reproducibility
previousRngState = rng(0,"twister");
% Load the trained agent from the .mat file
load('trained_agent.mat');
load('matlab.mat');
% load the necessary parameters into the base workspace in MATLAB
initializeRobotParameters;
%% Quadruped Robot Model

mdl = "rlQuadrupedRobot";
open_system(mdl)
%% RUN
function simResults = mysim(agent,env)
%% Simulate Trained Agent

% Fix the random stream for reproducibility.
 rng(0,"twister");
% To validate the performance of the trained agent, simulate it within the robot environment.
simOptions = rlSimulationOptions(MaxSteps=floor(Tf/Ts));
experience = sim(env,agent,simOptions);
% SIMRESULTS = mysim(AGENT,ENV) simulate AGENT within ENVIRONMENT
% with the simulation options specified on the Evaluate tab of the
% Reinforcement Learning Designer app and return simulation results SIMRESULTS.

% Reinforcement Learning Toolbox
% Generated on: 11-May-2025 00:29:17

%% Create simulation options
simOptions = rlSimulationOptions();
simOptions.NumSimulations = 10;

%% Perform simulations
simResults = sim(env,agent,simOptions);
end
%% plot time vs distance graph 
% Extract Time and Distance (X Position)
time = (0:size(simResults.Observation, 1) - 1) * env.Ts; % Adjust 'env.Ts' to your simulation time step
distance = simResults.Observation(:, 1); % Assuming the first observation is distance
% Plot Distance vs Time
figure;
plot(time, distance, 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Distance (m)');
title('Distance vs Time');
grid on;
