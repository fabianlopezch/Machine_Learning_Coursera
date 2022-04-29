function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% *********************** FEEDFORWARD CONST FUNCTION *********************
%                            (No regularization)


% Add ones to the X data matrix
X = [ones(m, 1) X];

% Computation of z for layer 2

z_2 = Theta1 * X';

% Activation nodes for layer 2

a_2 = sigmoid(z_2);

ones_to_add = ones(1,size(a_2, 2));

a_2 = [ones_to_add ; a_2];

% Computation of z for layer 3

z_3 = Theta2 * a_2;

% Activation nodes for layer 3

a_3 = sigmoid(z_3);

h = a_3; % This (k x m) matrix contains the values of the k-th output unit
         % for each m-th input sample

for i=1:m

    y_vec = zeros(num_labels,1); % Create the vector of labels
    y_vec(y(i)) = 1; % Set to one the position corresponding to the i-th input sample
    for k=1:num_labels
        % Cost fucntion computation
        J = J - y_vec(k) * log(h(k,i)) - (1 - y_vec(k)) * log(1 - h(k,i));
    end
end

J = J/m;

% ********************* REGULARIZED COST FUNCTION ************************

regTerm = 0;

for j=1:hidden_layer_size
    for k=2:input_layer_size+1
        regTerm = regTerm + Theta1(j,k)^2;
    end
end

for j=1:num_labels
    for k=2:hidden_layer_size+1
        regTerm = regTerm + Theta2(j,k)^2;
    end
end

regTerm = regTerm * (lambda/(2*m));

% Adding regularization term to J

J = J + regTerm;


% ************************ BACKPROPAGATION ******************************

Delta_1 = zeros(hidden_layer_size, input_layer_size + 1);
Delta_2 = zeros(num_labels, hidden_layer_size + 1);

for t=1:m

    % STEP 1: Perform feedforward pass for each t-th training example
    
    a_1 = X(t,:)';
    
    z_2 = Theta1 * a_1;     % Computation of z for layer    
    a_2 = sigmoid(z_2);     % Activation nodes for layer 2       
    a_2 = [1; a_2];         % Adding bias unit   
    z_3 = Theta2 * a_2;     % Computation of z for layer    
    a_3 = sigmoid(z_3);     % Activation nodes for layer 3 

    % STEP 2: Compute delta for the output layer

    y_vec = zeros(num_labels,1); % Create the vector of labels
    y_vec(y(t)) = 1; % Set to one the position corresponding to the i-th input sample

    delta_3 = a_3 - y_vec;

    % STEP 3: Compute delta for the hidden layer
    g_z2_prime = [1; sigmoidGradient(z_2)];
    delta_2 = Theta2' * delta_3 .* g_z2_prime;

    % STEP 4: Accumulate the gradation from the t-th example

    delta_2 = delta_2(2:end);

    Delta_2 = Delta_2 + delta_3 * a_2';
    Delta_1 = Delta_1 + delta_2 * a_1';

    % STEP 5: Obtain the (unregularized) gradient for the NN cost function

    Theta1_grad = (1/m) .* Delta_1;
    Theta2_grad = (1/m) .* Delta_2;

end





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
