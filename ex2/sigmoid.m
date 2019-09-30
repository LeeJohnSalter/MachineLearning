function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

%to ensure the process can cope with any type of data delivered (scalar/vector/matrix) 
%Determine the size of the delivered value of g and create an empty shell to update

g = zeros(size(z));

g = (g+1)./(1 + e.^-z);


% =============================================================

end
