function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. It is assumed that X has
%   the intercept term appended to the front of it.

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

%fprintf("Printing size of X: %d %d\n", size(X));
%fprintf("Printing size of theta: %d %d\n", size(theta));
hyp = sigmoid(X*theta);
pos_term = (y).*log(hyp);
neg_term = ((one-y)).*log(one-hyp);
lam_term = lambda/(2*m).*(sum(theta(2:end).^2));
J = (-1).*(m^(-1)).*sum(pos_term + neg_term) + lam_term;
grad(1) = (m^(-1)).*sum(((hyp - y)).*X(:, 1));
grad(2:end) = (m^(-1)).*sum(((hyp - y)).*X(:, 2:end));
lamb_mat = (lambda/m).*ones(size(grad(2:end)));
lamb_mat = lamb_mat.*theta(2:end);
grad(2:end) = grad(2:end) + lamb_mat;
% =============================================================

end
