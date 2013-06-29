function w = ridgereg(A, b, lamda)
% min ||Aw-b||_2 + ||lamda*w||_2

w = (A'*A+lamda*eye(size(A,2)))\(A'*b);