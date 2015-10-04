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

temp1=eye(num_labels);
yy=temp1(y,:);

xx=X;
X = [ones(m, 1) X];


part11=X*Theta1'; % z2 5000*25
part12=sigmoid(part11); %a2

part21=[ones(m, 1) part12];%a2 updated by additional column of ones
part22=sigmoid(part21*Theta2'); % einai to h_theta me vash to notation

part31=-yy.*log(part22);
part32=-(1-yy).*log(1-part22);
part33=part31+part32;

J1=(1/m)*sum(sum(part33));
part41=Theta1;
part41(:,1)=[];
part42=Theta2;
part42(:,1)=[]; % theta2 xwris thn prwth sthlh 10*25
J=J1+(lambda/(2*m))*(sum(sum(part41.*part41))+sum(sum(part42.*part42)));








%%%% for back propagation, check p. 7 and p. 13 at lecture 9
%%%% and https://class.coursera.org/ml-008/forum/thread?thread_id=1146
%%%% back propagation
d3=part22-yy;



d2=d3*part42.*sigmoidGradient(part11);



part41=[zeros(hidden_layer_size,1) part41];
Theta1_grad=(d2'*X)*(1/m)+(lambda/m)*part41;

part42=[zeros(num_labels,1) part42];
Theta2_grad=(d3'*part21)*(1/m)+(lambda/m)*part42;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
