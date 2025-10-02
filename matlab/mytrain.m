function [newAgent,trainStats] = mytrain(agent,env)
% [NEWAGENT,TRAINSTATS] = mytrain(AGENT,ENV) train AGENT within ENVIRONMENT
% with the training options specified on the Train tab of the Reinforcement Learning Designer app.
% mytrain returns trained agent NEWAGENT and training statistics TRAINSTATS.

% Reinforcement Learning Toolbox
% Generated on: 02-May-2025 00:31:43

%% Create training options
trainOptions = rlTrainingOptions();
trainOptions.MaxEpisodes = 10000;
trainOptions.ScoreAveragingWindowLength = 250;
trainOptions.StopTrainingValue = 300;
trainOptions.UseParallel = 1;
trainOptions.ParallelizationOptions.Mode = "async";
trainOptions.ParallelizationOptions.AttachedFiles = "";

%% Make copy of agent
newAgent = copy(agent);

%% Perform training
trainStats = train(newAgent,env,trainOptions);
