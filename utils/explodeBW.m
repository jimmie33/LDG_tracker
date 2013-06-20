%@I: single channel image 0-255
%@d: dimension of the vector
function [vector_field]=explodeBW(I,d)

vector_field=zeros(size(I,1),size(I,2),d);
step=255/d;
for i=1:d
    vector_field(:,:,i)=(I>(i-1)*step) & (I<=i*step);
end
