function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters. It is assumed that X has the intercept term
%   appended to the front of it.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
one = ones(size(y));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%fprintf("Printing size of X: %d %d\n", size(X));
%fprintf("Printing size of theta: %d %d\n", size(theta));
hyp = sigmoid(X*theta);
%fprintf("Printing size of hyp: %d %d\n", size(hyp));
%fprintf("Printing size of y: %d %d\n", size(y));
pos_term = (y).*log(hyp);
neg_term = ((one-y)).*log(one-hyp);
J = (-1).*(m^(-1)).*sum(pos_term + neg_term);
grad = (m^(-1)).*sum(((hyp - y)).*X);

% =============================================================

end
