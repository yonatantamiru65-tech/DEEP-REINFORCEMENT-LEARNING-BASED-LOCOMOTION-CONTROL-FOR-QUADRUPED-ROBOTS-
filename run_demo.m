%RUN_DEMO Simulate the exported trained TD3 policy; no training is started.
% Run this script from the repository root in MATLAB.
rng(0,'twister');
[env, agent, model] = setup_quadruped;
open_system(model);
sampleTime = evalin('base','Ts');
duration = evalin('base','Tf');
options = rlSimulationOptions('MaxSteps',ceil(duration/sampleTime), ...
    'NumSimulations',1,'UseExplorationPolicy',false);
experience = sim(env,agent,options);
disp('Simulation finished. Inspect experience and the model scopes.');
