function simResults = mysim(agent,env)
% SIMRESULTS = mysim(AGENT,ENV) simulate AGENT within ENVIRONMENT
% with the simulation options specified on the Evaluate tab of the
% Reinforcement Learning Designer app and return simulation results SIMRESULTS.

% Reinforcement Learning Toolbox
% Generated on: 16-Aug-2025 12:13:13

%% Create simulation options
simOptions = rlSimulationOptions();
simOptions.NumSimulations = 10;

%% Perform simulations
simResults = sim(env,agent,simOptions);
