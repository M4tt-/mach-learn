function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    % Compute gradient
    hyp = X*theta;                % hypothesis
    err = hyp - y;                % error
    
    %fprintf("Printing size of err: %d %d\n", size(err))
    %fprintf("Printing size of X: %d %d\n", size(X))
    gradient0 = (m^(-1)).*sum(err);        % Gradient for constant term
    %fprintf("Printing size of gradient0: %d %d\n", size(gradient0))
    gradient1 = (m^(-1)).*sum(err'*X(:,2));     % Gradient for first order term
    %fprintf("Printing size of gradient1: %d %d\n", size(gradient1))
    gradient = [gradient0; gradient1];
    %fprintf("Gradient: %f\n", gradient);
    
    % Update theta vectorally
    theta = theta - alpha.*gradient;
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    fprintf("Cost Function: %f\n", J_history(iter));

end

end
