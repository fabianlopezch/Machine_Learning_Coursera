function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% *********** Computing the cost function ***********
z = X*theta; 
h_theta = sigmoid(z); % Compute the hypothesys function value for each data sample
A = log(h_theta);
B = log(1 - h_theta);
A = -y .* A;
B = (1 - y) .* B;

J = sum(A - B) ./ m;

% *********** Computing the gradient ***********

for j=1:size(theta)
    temp = (h_theta - y) .* X(:,j);
    grad(j) = sum(temp) / m;
end




% =============================================================

end
