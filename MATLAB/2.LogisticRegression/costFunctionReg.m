function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

z = X*theta; 
h_theta = sigmoid(z); % Compute the hypothesys function value for each data sample
A = log(h_theta);
B = log(1 - h_theta);
A = -y .* A;
B = (1 - y) .* B;

J = sum(A - B) ./ m;

% ****** Adding regularization term

Reg_term = (lambda/(2*m)) * ((theta' * theta) - (theta(1)^2)); 
J = J + Reg_term;


% *********** Computing the gradient ***********

for j=1:size(theta)
    if j==1
        temp = (h_theta - y) .* X(:,j);
        grad(j) = sum(temp) / m;
    else
        temp = (h_theta - y) .* X(:,j);
        grad(j) = sum(temp) / m;
        grad(j) = grad(j) + (lambda/m)*theta(j);
    end
end



% =============================================================

end
