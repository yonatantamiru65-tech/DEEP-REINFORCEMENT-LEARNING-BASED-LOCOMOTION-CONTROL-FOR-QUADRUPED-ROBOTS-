function simResults = mysim(agent,env)
% SIMRESULTS = mysim(AGENT,ENV) simulate AGENT within ENVIRONMENT
% with the simulation options specified on the Evaluate tab of the
% Reinforcement Learning Designer app and return simulation results SIMRESULTS.

% Reinforcement Learning Toolbox
% Generated on: 13-May-2025 21:47:00

%% Create simulation options
simOptions = rlSimulationOptions();
simOptions.MaxSteps = 200;
simOptions.NumSimulations = 10000;

%% Perform simulations
simResults = sim(env,agent,simOptions);
