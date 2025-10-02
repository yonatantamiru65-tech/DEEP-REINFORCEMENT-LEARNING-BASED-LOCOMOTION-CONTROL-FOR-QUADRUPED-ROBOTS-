  numObs = 44;
numAct = 8;
function [criticNet1, criticNet2, actorNet] = createQuadNetworks(numObs, numAct)

% Critic Network 1
statePath = featureInputLayer(numObs, 'Name','observation');
actionPath = featureInputLayer(numAct, 'Name','action');
commonPath = [
    concatenationLayer(1,2,'Name','concat')
    fullyConnectedLayer(400,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(300,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(1,'Name','CriticOutput')];
criticNet1 = layerGraph(statePath);
criticNet1 = addLayers(criticNet1, actionPath);
criticNet1 = addLayers(criticNet1, commonPath);
criticNet1 = connectLayers(criticNet1, 'observation', 'concat/in1');
criticNet1 = connectLayers(criticNet1, 'action', 'concat/in2');

% Critic Network 2 (identical structure, separate parameters)
criticNet2 = criticNet1;

% Actor Network
actorNet = [
    featureInputLayer(numObs,'Name','observation')
    fullyConnectedLayer(400,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(300,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(numAct,'Name','act')
    tanhLayer('Name','tanh')]; % scale to [-1, 1]

end
