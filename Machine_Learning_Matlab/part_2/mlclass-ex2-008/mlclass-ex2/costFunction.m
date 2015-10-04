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


mults=X*theta;
lmults1=log(sigmoid(mults));
lmults2=log(1-sigmoid(mults));

part1=-y.*lmults1;
part2=-(1-y).*lmults2;

J=(1/m)*sum(part1+part2);

grad(1)=(1/m)*sum((sigmoid(mults)-y).*X(:,1));
grad(2)=(1/m)*sum((sigmoid(mults)-y).*X(:,2));
grad(3)=(1/m)*sum((sigmoid(mults)-y).*X(:,3));



% =============================================================

end
