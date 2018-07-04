function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

m = length(y);                 % number of training examples
hyp = X*theta;                 % hypothesis
err = hyp - y;                 % error
J = ((2*m)^-(1))*sum(err.^2);  % return, cost function

end
