%@I: single channel image
%@d: vecotr dimension
%@filter_sz: 3d filter size
function [vector_field]=explodeBlur(I,d,filter_sz,sigma) 
vector_field = explodeBW(I,d);
%vector_field = imdilate(vector_field,strel('arbitrary',ones(floor(filter_sz./2))));
vector_field = imfilter(vector_field,fspecial('average',filter_sz),'same','replicate');