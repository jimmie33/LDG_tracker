function [feat F] = getFeatureRep(I,alpha,nbin,k,pixel_step)
% compute feature representation: mxnxd, d is the feature dimension
% decay factor and nbin is for the local histogram computation
%
global config
fd = config.fd;
thr = repmat(reshape(config.thr,1,1,[]),[size(I,1), size(I,2)]);

tic

hist_mtx1 = LSH(I(:,:,1)*255, alpha, nbin);%local histogram 0.033

toc

tic
F{1} = IIF(I(:,:,1)*255, hist_mtx1, k, nbin);%semi-affine invariant feature
F{2} = IIF2(I(:,:,1)*255, hist_mtx1, k, nbin);%feature by pixel ordering
F{3} = I(:,:,1);%gray image
if config.use_color
    F{4} = I(:,:,2);%color part
    F{5} = I(:,:,3);%color part
end



if ~config.use_color
    feat = zeros([size(I(:,:,1)),3*config.fd]);
else
    feat = zeros([size(I(:,:,1)),5*config.fd]);
end

toc

tic


for i = 1:numel(F)
    feat(:,:,(i-1)*fd+1:i*fd) = repmat(F{i},[1 1 fd]) > thr;
end
feat = imfilter(feat,fspecial('gaussian',[9 9],0.5*pixel_step),'same','replicate');

toc
