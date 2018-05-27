function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Create a vector of candidate C and sigma values
C_cand = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 90];
sigma_cand = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 90];

% Create aplaceholder forsmallest prediction error
min_pred_err = 10;
min_C = C_cand(1);
min_sigma = sigma_cand(1);

% We need to test many values of C and sigma. Nested for loop ...
count = 0;
for i=1:length(C_cand)
    % Try a candidate C value
    C_test = C_cand(i);
    
    % Loop over all candidate sigma values
    for j=1:length(sigma_cand)
        % Try a candidate sigma value
        sigma_test = sigma_cand(j);
        
        % Get a model for the training set
        model = svmTrain(X, y, C_test, ...
            @(x1, x2) gaussianKernel(x1, x2, sigma_test));
        
        % Get predictions for the trained model
        predictions = svmPredict(model, Xval);

        % Compute the prediction error
        pred_error = mean(double(predictions ~= yval));
        
        % Retain choice of sigma and C if current pred. error is best so
        % far
        if pred_error < min_pred_err
           min_C = C_test;
           min_sigma = sigma_test;
           min_pred_err = pred_error;
        end
    end
    count = count + 1;
end

C = min_C;
sigma = min_sigma;

% =========================================================================

end
