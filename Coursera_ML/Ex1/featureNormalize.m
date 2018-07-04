function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
       
for k = 1:length(X(1, :))
   mu(k) = mean(X(:, k));
   sigma(k) = std(X(:, k));
   X_norm(:, k) = (X(:, k) - mu(k))./(sigma(k));
end

end
