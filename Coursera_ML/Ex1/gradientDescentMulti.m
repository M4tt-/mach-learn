function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y);          % number of training examples
%n = size(X, 2);         % number of features in feature space
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % Compute gradient
    hyp = X*theta;                % hypothesis
    err = hyp - y;                % error
    gradient = ((m^(-1))*(err')*X)';
    %{
    fprintf("Printing size of err: %d %d\n", size(err))
    fprintf("Printing size of X: %d %d\n", size(X))
    fprintf("Printing size of theta: %d %d\n", size(theta));
    fprintf("Printing size of gradient: %d %d\n", size(gradient));
    %}
    
    % Update theta vectorally
    theta = theta - alpha.*gradient;

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
