function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

m = length(y);                            % number of training examples
hyp = X*theta;                            % hypothesis
err = hyp - y;                            % error
J = ((2*m)^-(1))*(err')*err;              % return, cost function
%fprintf("Cost function: %f\n", J)
end
