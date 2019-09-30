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

%work out our jypothesis for current theta values;
h = X*theta;

%Ensure theta0 has a value of 0 so is ignored from the cost function;
l=ones(length(theta),1);
l(1)=0;
theta_reg=(theta.*l);


J = (1/(2*m))*sum((h-y).^2)+(lambda/(2*m))*theta_reg'*theta_reg;

grad = (1/m)*(X'*(h-y)+lambda*theta_reg);

% =========================================================================

grad = grad(:);

end
