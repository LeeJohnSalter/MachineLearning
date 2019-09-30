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


%first calculate the sigmoid function for the parameter estimates;
sig = sigmoid(X*theta);

% LS: Note -  Multiplied out the negative number to give a simpler  looking J value;
% LS: Also ensure the y values are transposed to ensure they are the correct dimension
% LS: to multiply against the returned sig matrix;
% LS: Now need to add the lamda adjustmented ignoring this of theta 0;
% Attempting to do this through a matrix of 0 for the theta 0 terms and 1 for the remaining


l=ones(length(theta),1);
l(1)=0;

%Ensure theta0 has a value of 0 so is ignored from the cost function;
adjTheta=(theta.*l);

J = (1/m)*(-y'* log(sig) - (1 - y)'* log(1-sig)) + (lambda/(2*m)).*(theta'*adjTheta);

% LS: gradion needs to include the partial derivative of the lamda adjustment l/m.0^2 
% LS: which is 2l0 .  Note 2 cancels out with the 1/2m;

grad = (1/m)* (X'*(sig - y) + lambda*adjTheta);





% =============================================================

end
