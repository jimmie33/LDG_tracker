function [feat F] = getFeatureRep(I,alpha,nbin,k,pixel_step)
% compute feature representation: mxnxd, d is the feature dimension
% decay factor and nbin is for the local histogram computation
%
global config

if size(I,3) == 1
    I = double(I)/255;
    config.use_color = false;
elseif config.use_color
%     I = cv.cvtColor(I,'RGB2Lab');
%     I = double(I)/255;
    I = RGB2Lab(I);
else
    I = double(rgb2gray(I))/255;
end

fd = config.fd;
thr = repmatls(reshape(config.thr,1,1,[]),[size(I,1), size(I,2)]);

hist_mtx1 = LSH(I(:,:,1)*255, alpha, nbin);%local histogram 0.033

% F{1} = IIF(I(:,:,1)*255, hist_mtx1, k, nbin);%semi-affine invariant feature
F{1} = IIF2(I(:,:,1)*255, hist_mtx1, k, nbin);%feature by pixel ordering
F{2} = I(:,:,1);%gray image
if config.use_color
    F{3} = I(:,:,2);%color part
    F{4} = I(:,:,3);%color part
end



if config.use_raw_feat
    feat = reshape(cell2mat(F),size(F{1},1),size(F{1},2),[]);    
else
    if ~config.use_color
        feat = zeros([size(I(:,:,1)),2*config.fd]);
    else
        feat = zeros([size(I(:,:,1)),4*config.fd]);
    end
    for i = 1:numel(F)
        feat(:,:,(i-1)*fd+1:i*fd) = repmatls(F{i},[1 1 fd]) > thr;
    end    
end

feat = imfilter(feat,fspecial('gaussian',[9 9],0.2*pixel_step),'same','replicate');

