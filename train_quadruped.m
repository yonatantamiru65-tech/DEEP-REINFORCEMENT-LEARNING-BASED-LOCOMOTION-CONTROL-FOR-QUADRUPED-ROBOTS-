function [agent, stats] = train_quadruped(algorithm, maxEpisodes)
%TRAIN_QUADRUPED Train a fresh TD3/DDPG policy using the supplied robot model.
% Example: [agent,stats] = train_quadruped('td3',1000);
% These launch settings are a reproducible adapter, not an exact replay of
% every historical Designer session. The original sessions are release assets.
if nargin < 1, algorithm = 'td3'; end
if nargin < 2, maxEpisodes = 1000; end
validateattributes(maxEpisodes,{'numeric'},{'scalar','integer','positive'});
algorithm = validatestring(lower(algorithm),{'td3','ddpg'});
rng(0,'twister');
[env, referenceAgent, model] = setup_quadruped;
obsInfo = getObservationInfo(referenceAgent);
actInfo = getActionInfo(referenceAgent);
sampleTime = evalin('base','Ts');
if strcmp(algorithm,'td3')
    agent = createTD3Agent(prod(obsInfo.Dimension),obsInfo, ...
        prod(actInfo.Dimension),actInfo,sampleTime);
else
    [criticNet,~,actorNet] = createQuadNetworks( ...
        prod(obsInfo.Dimension),prod(actInfo.Dimension));
    actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);
    critic = rlQValueFunction(criticNet,obsInfo,actInfo, ...
        'ObservationInputNames','observation','ActionInputNames','action');
    options = rlDDPGAgentOptions('SampleTime',sampleTime, ...
        'MiniBatchSize',256,'ExperienceBufferLength',1e6);
    options.ActorOptimizerOptions.LearnRate = 1e-4;
    options.CriticOptimizerOptions.LearnRate = 1e-3;
    agent = rlDDPGAgent(actor,critic,options);
end
assignin('base','agent1_Trained',agent);
set_param([model '/RL Agent'],'Agent','agent1_Trained');
options = rlTrainingOptions('MaxEpisodes',maxEpisodes, ...
    'MaxStepsPerEpisode',ceil(evalin('base','Tf')/sampleTime), ...
    'ScoreAveragingWindowLength',250,'UseParallel',false, ...
    'Plots','training-progress');
stats = train(agent,env,options);
root = fileparts(mfilename('fullpath'));
output = fullfile(root,'generated');
if ~isfolder(output), mkdir(output); end
agent1_Trained = agent; %#ok<NASGU>
save(fullfile(output,[algorithm '_agent.mat']),'agent1_Trained','stats','-v7.3');
end
