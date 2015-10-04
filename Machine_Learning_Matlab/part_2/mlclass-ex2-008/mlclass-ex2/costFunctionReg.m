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



mults=X*theta;
lmults1=log(sigmoid(mults));
lmults2=log(1-sigmoid(mults));

part1=-y.*lmults1;
part2=-(1-y).*lmults2;

theta1=theta(2:length(theta));

J=(1/m)*sum(part1+part2)+(lambda/m/2)*sum(theta1.*theta1);
%J=(1/m)*sum(part1+part2);


grad(1)=(1/m)*sum((sigmoid(mults)-y).*X(:,1));

for i=2:length(theta)
    
    grad(i)=(1/m)*sum((sigmoid(mults)-y).*X(:,i))+lambda*theta(i)/m;

end


%     z = X * theta;
%     h = sigmoid(z);
%     
%     
%     J1=(1/m)*sum(part1+part2);
%       J =J1+ ...
%     (lambda/(2*m))*norm(theta([2:end]))^2;
%     
% J=(1/m)*sum(part1+part2)+(lambda/m/2)*sum(theta1.*theta1);
%     % Calculate J (for testing convergence)
% %     J =(1/m)*sum(-y.*log(h) - (1-y).*log(1-h))+ ...
% %     (lambda/(2*m))*norm(theta([2:end]))^2;
% %     
%     % Calculate gradient and hessian.
%     G = (lambda/m).*theta; G(1) = 0; % extra term for gradient
%     %L = (lambda/m).*eye(n); L(1) = 0;% extra term for Hessian
%     grad = ((1/m).*X' * (h-y)) + G;





% =============================================================

end
