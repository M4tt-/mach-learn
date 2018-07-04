function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations. You may need to add a column
%   of 1s to X to accont for the intercept term.

theta = pinv(X'*X)*X'*y;

end
