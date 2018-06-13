function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% Get dimensionality of training examples
n = size(X,2);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);
% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
for i=1:length(idx)

    % Vectorize the training example
    train_ex = repmat(X(i,:), K, 1);

    % Calculate the difference vector between each centroid and train_ex
    diff_vec = train_ex - centroids;

    % Calculate the Euclidean norm of each difference vector
    distances = vecnorm(diff_vec, 2, 2).^2

    % Select the index of the minimum distance, i.e. 'j'
    [m, j] = min(distances);

    % Assign the index to idx
    idx(i) = j;
end

% =============================================================

end

