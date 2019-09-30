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


%first calculate the sigmoid function for the parameter estimates;
sig = sigmoid(X*theta);

% LS: Note -  Multiplied out the negative number to give a simpler  looking J value;
% LS: Also ensure the y values are transposed to ensure they are the correct dimension
% LS: to multiply against the returned sig matrix;

J = (1/m)*(-y'* log(sig) - (1 - y)'* log(1-sig));

grad = (1/m)*X'*(sig - y);


% =============================================================

end
