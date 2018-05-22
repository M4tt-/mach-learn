function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad.
%   Assumes that X has the bias unit already appended to it.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Define the hypothesis
hyp = X*theta;

% Compute the squared error term
err = (hyp - y).^2;

% Compute the unregularized cost
J0 = 1/(2*m)*sum(sum(err));

% Compute the regularization term
theta_R = theta;
theta_R(1) = 0;       % Ensure the bias term doesn't contribute to the sum
JR = lambda/(2*m)*sum(theta_R.^2);

% Total cost
J = J0 + JR;

% Compute the gradient ----------------------------------------------------
grad = (1/m).*(hyp - y)'*X;

% Incorporate regularization on every term except the bias term
grad_R = (lambda/m).*theta';
grad_R(:, 1) = 0;
grad = grad + grad_R;

% =========================================================================

grad = grad(:);

end
