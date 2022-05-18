function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

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


h = X * theta;          % Evaluating the hypothesis function h(x)
h_y = h-y;              % Difference 
J = h_y.^2;             % Computing the squared error
J = (1/(2*m)) * sum(J); % Summing up the errors and multiplying by (1/2m)
Reg = theta(2:end)' * theta(2:end); % Computing the summation for the regularization term using dot product and discarding theta_0
Reg = (lambda/(2*m)) * Reg;

J = J + Reg;    % Adding the regularization term to our previously computed cost


% *********** Computing the gradient ***********

for j=1:size(theta)
    if j==1     % ommit regularization for theta_0
        temp = h_y .* X(:,j);
        grad(j) = sum(temp) / m;
    else
        temp = h_y .* X(:,j);
        grad(j) = sum(temp) / m;
        grad(j) = grad(j) + (lambda/m)*theta(j);
    end
end


% =========================================================================

grad = grad(:);

end
