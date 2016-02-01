function [pred] = stackedAEPredict_bak(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

M = size(data, 2);

%% feed forward

% MLP
numLayers = numel(stack);
inputs = cell(numLayers+1,1);
inputs{1} = data;
for d = 1:numLayers
    z = stack{d}.w * inputs{d} + repmat(stack{d}.b, [1,M]);
    a = sigmoid(z);
    inputs{d+1} = a;
end

% Softmax
x = inputs{numLayers+1};
z = softmaxTheta * x;
ez = exp(z);
ez_sum = sum(ez,1);
h = ez ./ repmat(ez_sum,[numClasses,1]);

[test, pred] = max(h,[],1);









% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
