function agent = createTD3Agent(numObs, obsInfo, numAct, actInfo, Ts)
% TD3 agent for Quadruped Robot with 44 obs and 8 actions

%% Create custom networks for quadruped
[criticNetwork1, criticNetwork2, actorNetwork] = createQuadNetworks(numObs, numAct);

%% Optimizer Options
criticOptions = rlOptimizerOptions('Optimizer','adam','LearnRate',1e-3,'GradientThreshold',1);
actorOptions  = rlOptimizerOptions('Optimizer','adam','LearnRate',1e-4,'GradientThreshold',1);

%% Create Actor and Critic Representations
critic1 = rlQValueFunction(criticNetwork1, obsInfo, actInfo, ...
    'ObservationInputNames','observation','ActionInputNames','action');
critic2 = rlQValueFunction(criticNetwork2, obsInfo, actInfo, ...
    'ObservationInputNames','observation','ActionInputNames','action');
actor = rlContinuousDeterministicActor(actorNetwork, obsInfo, actInfo);

%% TD3 Agent Options
agentOptions = rlTD3AgentOptions;
agentOptions.SampleTime = Ts;
agentOptions.DiscountFactor = 0.99;
agentOptions.MiniBatchSize = 256;
agentOptions.ExperienceBufferLength = 1e6;
agentOptions.TargetSmoothFactor = 5e-3;
agentOptions.NumEpoch = 3;
agentOptions.MaxMiniBatchPerEpoch = 100;
agentOptions.LearningFrequency = -1;
agentOptions.PolicyUpdateFrequency = 2;
agentOptions.TargetUpdateFrequency = 2;

% Target Policy Smoothing
agentOptions.TargetPolicySmoothModel.StandardDeviation = 0.2;
agentOptions.TargetPolicySmoothModel.StandardDeviationMin = 0.1;
agentOptions.TargetPolicySmoothModel.LowerLimit = -0.5;
agentOptions.TargetPolicySmoothModel.UpperLimit = 0.5;

% Exploration Model
agentOptions.ExplorationModel = rl.option.OrnsteinUhlenbeckActionNoise;
agentOptions.ExplorationModel.MeanAttractionConstant = 1;
agentOptions.ExplorationModel.StandardDeviation = 0.1;

agentOptions.ActorOptimizerOptions = actorOptions;
agentOptions.CriticOptimizerOptions = criticOptions;

%% Final Agent
agent = rlTD3Agent(actor, [critic1, critic2], agentOptions);
end
