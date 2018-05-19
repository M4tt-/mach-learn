function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
%   A column of ones is added to the input vector X for the bias units.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add a column of ones to the input vector X
X0 = [ones(m, 1), X];  % This is used as "a1" as per tutorial for backprop

% Create the 'one vector'
one = ones(1, num_labels);

% Compute the activation units for layer 1
a1 = sigmoid(X0*Theta1');

% Add ones to the a1 matrix
a1 = [ones(m,1), a1];
a2 = a1;

% Compute the output layer
a2_f = sigmoid(Theta2*a1');

% The output layer is our hypothesis
hyp = a2_f';

pos_term = zeros(size(hyp));
neg_term = zeros(size(hyp));
for i=1:m

    % Transform y to the appropriate vector for multi-class classification
    y_vec = zeros(size(hyp(i, :)));
    idx = y(i);
    y_vec(idx) = 1;

    % Compute the cost function for each training example
    pos_term(i, :) = -y_vec.*log(hyp(i, :));
    neg_term(i, :) = (one - y_vec).*log(one - hyp(i, :));

end

J0 = (m)^(-1)*sum(sum(pos_term - neg_term));

% Compute the regularization term. Disregard the bias term in each 
% Theta matrix (assumed to be the first column)
R1 = sum(sum(Theta1(:,2:end).^2));
R2 = sum(sum(Theta2(:,2:end).^2));
R = lambda/(2*m)*(R1+R2);

% Total cost with regularization
J = J0 + R;
% -------------------------------------------------------------

% Backpropagation 
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

% Transform y to be vectorized for ulti-class classification
y_mat = zeros(size(m, num_labels));
for j=1:length(y)
   idx = y(j);
   y_mat(j, idx) = 1;
end

% Compute delta^(3)
a3 = hyp;
d3 = a3 - y_mat;

% Compute delta^(2)
d2_0 = d3*Theta2;
d2 = d2_0.*(a1.*(1-a1));
d2 = d2(:, 2:end);

% Construct the unscaled derivative matrix

Delta1 = d2'*X0;
Delta2 = d3'*a2;

% Compute the regularization term for the hidden layer
Delta1_R = zeros(size(Delta1));
Delta2_R = zeros(size(Delta2));
Theta1_R = (lambda/m).*Theta1;
Theta2_R = (lambda/m).*Theta2;
Theta1_R(:, 1) = 0;
Theta2_R(:, 1) = 0;

Theta1_grad = Delta1./m + Theta1_R;
Theta2_grad = Delta2./m + Theta2_R;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
