% Fix Random Seed Generator to Improve Reproducibility
previousRngState = rng(0,"twister");

% load the necessary parameters into the base workspace in MATLAB
initializeRobotParameters;
%% Quadruped Robot Model

mdl = "rlQuadrupedRobot";
open_system(mdl)

% Load the trained agent from the .mat file
load('trained_agent.mat');
load('matlab.mat');
%% Create Environment Object

% Specify the parameters for the observation set.
numObs = 44;
obsInfo = rlNumericSpec([numObs 1]);
obsInfo.Name = "observations";

 % Specify the parameters for the action set.
numAct = 8;
actInfo = rlNumericSpec([numAct 1],LowerLimit=-1,UpperLimit=1);
actInfo.Name = "torque";

 % Create the environment using the reinforcement learning model.
blk = mdl + "/RL Agent";
env = rlSimulinkEnv(mdl,blk,obsInfo,actInfo);

 % introduces random deviations into the initial joint angles and angular velocities.
env.ResetFcn = @quadrupedResetFcn;

%% |Create TD3 Agent Object
rng(0,"twister");
% Create the TD3 agent options
agentOptions = rlTD3AgentOptions();

% Specify the options
agentOptions.SampleTime = Ts;   % Sample time
agentOptions.MiniBatchSize = 256;   % Mini batch size
agentOptions.ExperienceBufferLength = 1e6;  % Experience buffer length
agentOptions.MaxMiniBatchPerEpoch = 200;   % Max mini-batches per epoch

% Optimizer options (same as DDPG)
agentOptions.ActorOptimizerOptions.LearnRate = 1e-3;
agentOptions.CriticOptimizerOptions.LearnRate = 1e-3;

% Exploration options (same as DDPG)
agentOptions.NoiseOptions.StandardDeviation = 0.1;
initOpts = rlAgentInitializationOptions(NumHiddenUnit=256);
agent = createTD3Agent(obsInfo, actInfo, initOpts, agentOptions);

%%% *% Set the agent in Simulink model
set_param('rlQuadrupedRobot/RL Agent', 'Agent', 'agent');
actor = getActor(agent);
critic = getCritic(agent);
actorNet = getModel(actor);
criticNet = getModel(critic(1));
summary(actorNet);
plot(actorNet);
summary(criticNet);
plot(criticNet);
%% Specify Training Options

trainOpts = rlTrainingOptions(...
    MaxEpisodes=100,...
    trainOptsMaxStepsPerEpisode=100,...
    MaxStepsPerEpisode=floor(Tf/Ts),...
    ScoreAveragingWindowLength=250,...
    Plots="training-progress",...
    StopTrainingCriteria="EvaluationStatistic",...
    StopTrainingValue=300);

%  train the agent in parallel
trainOpts.UseParallel = true;
trainOpts.ParallelizationOptions.Mode = "async";
%% Train Agent

% Fix the random stream for reproducibility.
rng(0,"twister");

% Set the doTraining flag
doTraining = false;
if doTraining      
    % Train the agent.
    %result = train(agent, env, trainOpts);
    result = train(agent, env, trainOpts, Evaluator=evaluator);
else
    % Load pretrained agent parameters for the example, if the file exists
        load("rlQuadrupedAgentParams.mat","params")
    setLearnableParameters(agent, params);
    %pretrainedFile = "rlQuadrupedAgentParams.mat";
    % if exist(pretrainedFile, 'file')  % Check if the file exists before loading
    %     load(pretrainedFile, "params");
    % else
    %     load(pretrainedFile, "params");
    % end
end

%% Simulate Trained Agent

% Fix the random stream for reproducibility.
 rng(0,"twister");
% To validate the performance of the trained agent, simulate it within the robot environment.
AgentOption = rlSimulationOptions(MaxSteps=floor(Tf/Ts));
experience = sim(env,agent,AgentOption);
%% result
% Performance Metrics for Robot Locomotion (Robotics Perspective)
% Example Data for Simulation (You can replace these with real data)
time = 0:1:10; % Time (seconds) for 10-second simulation
distance_travelled = cumsum(rand(1,10)*0.5); % Simulated distance (meters)
speed = diff([0, distance_travelled]); % Speed (m/s), using diff to get speed between steps
energy_consumed = cumsum(rand(1,10)*10); % Simulated energy (Joules)

% Roll, Pitch, and Yaw angles (random data, replace with actual values)
roll = rand(1,10) * 10 - 5; % Roll angles (-5 to 5 degrees)
pitch = rand(1,10) * 10 - 5; % Pitch angles (-5 to 5 degrees)
yaw = rand(1,10) * 20 - 10; % Yaw angles (-10 to 10 degrees)

% Efficiency (Energy consumption per unit distance)
efficiency = energy_consumed ./ distance_travelled; 

% 1. Plot Speed (meters per second)
figure;
plot(time(2:end), speed, 'LineWidth', 2);
title('Robot Speed vs Time');
xlabel('Time (seconds)');
ylabel('Speed (m/s)');
grid on;
% Display the plot
disp('Displaying Speed vs Time plot...');
shg;  % Bring figure window to front
% Save the plot as an image
saveas(gcf, 'robot_speed_vs_time.png'); 

% 2. Plot Efficiency (Energy consumption per unit distance)
figure;
plot(distance_travelled, efficiency, 'LineWidth', 2);
title('Energy Efficiency vs Distance');
xlabel('Distance (meters)');
ylabel('Energy Efficiency (Joules/m)');
grid on;
% Display the plot
disp('Displaying Energy Efficiency vs Distance plot...');
shg;  % Bring figure window to front
% Save the plot as an image
saveas(gcf, 'energy_efficiency_vs_distance.png'); 

% 3. Plot Stability (Roll, Pitch, and Yaw)
figure;
subplot(3,1,1);
plot(time, roll, 'r-', 'LineWidth', 2);
title('Roll Angle vs Time');
xlabel('Time (seconds)');
ylabel('Roll Angle (degrees)');
grid on;

subplot(3,1,2);
plot(time, pitch, 'b-', 'LineWidth', 2);
title('Pitch Angle vs Time');
xlabel('Time (seconds)');
ylabel('Pitch Angle (degrees)');
grid on;

subplot(3,1,3);
plot(time, yaw, 'g-', 'LineWidth', 2);
title('Yaw Angle vs Time');
xlabel('Time (seconds)');
ylabel('Yaw Angle (degrees)');
grid on;
% Display the plot
disp('Displaying Robot Stability vs Time plot...');
shg;  % Bring figure window to front
% Save the plot as an image
saveas(gcf, 'robot_stability_vs_time.png'); 
%% data logger
% Create a DataLogger object
datalogger = rlDataLogger;

% Set simulation options with the datalogger
simOptions = rlSimulationOptions('MaxSteps', 500, 'DataLogger', datalogger);

% Run the simulation with logging enabled
experience = sim(env, agent, simOptions);

% Extract the logged data (e.g., rewards)
loggedData = datalogger.LoggingInfo;  % Logging data
rewards = loggedData.Reward;  % Extract rewards data
steps = 1:length(rewards);  % Create steps vector (time steps)

% Plot the rewards over time
figure;
plot(steps, rewards);
xlabel('Steps');
ylabel('Reward');
title('Agent Rewards over Time');
grid on;
%%
%% Extract rewards
% Extract rewards from the buffer
episodeRewards = sum(reward, 2); % Sum of rewards for each episode

% Plotting
figure;
plot(episodeRewards);
title('Episode Rewards Over Time');
xlabel('Episode');
ylabel('Total Reward');
grid on;
%% plot time vs distance graph 
% Initialize the logger
logger = rlDataLogger();
logDir = fullfile(pwd, "myDataLog");
logger.LoggingOptions.LoggingDirectory = logDir;
logger.LoggingOptions.FileNameRule = "episode<id>";

% Attach callback functions
logger.AgentLearnFinishedFcn = @agentLearnFinishedFcn;
logger.AgentStepFinishedFcn  = @agentStepFinishedFcn;
logger.EpisodeFinishedFcn    = @episodeFinishedFcn;

function dataToLog = agentLearnFinishedFcn(data)
    % Logs the loss values after each learning step
    dataToLog.ActorLoss = data.ActorLoss;
    dataToLog.CriticLoss = data.CriticLoss;
    dataToLog.TDError = data.TDError;
end

function dataToLog = agentStepFinishedFcn(data)
    % Logs the noise type applied during exploration
    policy = getExplorationPolicy(data.Agent);
    dataToLog.NoiseType = policy.NoiseType;
end

function dataToLog = episodeFinishedFcn(data)
    % Logs cumulative reward and Q-value estimates at the end of each episode
    dataToLog.Experience = data.Experience;
    dataToLog.EpisodeReward = data.EpisodeInfo.CumulativeReward;
    if data.EpisodeInfo.StepsTaken > 0
        dataToLog.EpisodeQ0 = evaluateQ0(data.Agent, data.EpisodeInfo.InitialObservation);
    else
        dataToLog.EpisodeQ0 = 0;
    end
end

% Run the simulation with logging
simOptions = rlSimulationOptions('MaxSteps', 1000);
simResults = sim(env, agent, simOptions);

% Save the logger data
save(fullfile(logDir, 'simulation_logs.mat'), 'logger');
% Load logged data
load(fullfile(logDir, 'simulation_logs.mat'), 'logger');

% Extract rewards
rewards = [logger.EpisodeFinishedFcn.EpisodeReward];

% Plot
figure;
plot(rewards, 'LineWidth', 1.5);
xlabel('Episode Number');
ylabel('Episode Reward');
title('Episode Reward vs. Episode Number');
grid on;

