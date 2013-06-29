function K = linear_kernel(u,v,varargin) 
%LINEAR_KERNEL Linear kernel for SVM functions

% Copyright 2004-2010 The MathWorks, Inc.
% $Revision: 1.1.12.3.14.2 $  $Date: 2011/03/17 22:25:58 $
K = (u*v');