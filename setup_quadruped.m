function [env, agent, model] = setup_quadruped(agentFile)
%SETUP_QUADRUPED Load the supplied robot, parameters and exported TD3 policy.
% This setup adapter does not train or overwrite the supplied model/agent.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'simulation'),fullfile(root,'src'));
if nargin == 0
    agentFile = fullfile(root,'agents','trained_agent.mat');
end
assert(isfile(agentFile),'quadruped:MissingAgent', ...
    'Cannot find trained agent: %s',agentFile);
parameterFile = fullfile(root,'simulation','initializeRobotParameters.m');
evalin('base',sprintf('run(''%s'');',strrep(parameterFile,'''','''''')));
saved = load(agentFile,'agent1_Trained');
assert(isfield(saved,'agent1_Trained'),'quadruped:AgentVariable', ...
    'The selected file must contain agent1_Trained.');
agent = saved.agent1_Trained;
assignin('base','agent1_Trained',agent);
model = 'rlQuadrupedRobot';
load_system(fullfile(root,'simulation',[model '.slx']));
set_param([model '/RL Agent'],'Agent','agent1_Trained');
obsInfo = getObservationInfo(agent);
actInfo = getActionInfo(agent);
env = rlSimulinkEnv(model,[model '/RL Agent'],obsInfo,actInfo);
env.ResetFcn = @quadrupedResetFcn;
end
