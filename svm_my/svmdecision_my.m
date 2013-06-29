function [out,f] = svmdecision_my(Xnew,svm_struct)
%SVMDECISION Evaluates the SVM decision function

%   Copyright 2004-2010 The MathWorks, Inc.
%   $Revision: 1.1.12.4.28.2 $  $Date: 2011/03/17 22:26:00 $

% sv = svm_struct.SupportVectors;
% alphaHat = svm_struct.Alpha;
% bias = svm_struct.Bias;
% kfun = svm_struct.KernelFunction;
% kfunargs = svm_struct.KernelFunctionArgs;
f = Xnew*svm_struct.w';
out = sign(f);
% points on the boundary are assigned to class 1
out(out==0) = 1;