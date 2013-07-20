function [feat F] = getFeatureRep(I,alpha,nbin,k,pixel_step)
% compute feature representation: mxnxd, d is the feature dimension
% decay factor and nbin is for the local histogram computation
%
global config

if size(I,3) == 3 && config.use_color
%     I = cv.cvtColor(I,'RGB2Lab');
%     I = double(I)/255;
    I = RGB2Lab(I);
elseif size(I,3) == 3
    I = rgb2gray(I);
end

fd = config.fd;
thr = repmatls(reshape(config.thr,1,1,[]),[size(I,1), size(I,2)]);

% hist_mtx1 = LSH(I(:,:,1)*255, config.pixel_step*4+1, nbin);%local histogram 0.033

% F{1} = IIF(I(:,:,1)*255, hist_mtx1, k, nbin);%semi-affine invariant feature
ksize = config.pixel_step*4+1;
F{1} = 255-cv.calcIIF(I(:,:,1),[ksize ksize],nbin);%IIF2(I(:,:,1)*255, hist_mtx1, nbin);%feature by pixel ordering
F{2} = I(:,:,1);%gray image
if config.use_color
    F{3} = I(:,:,2);%color part
    F{4} = I(:,:,3);%color part
end



if config.use_raw_feat
    feat = double(reshape(cell2mat(F),size(F{1},1),size(F{1},2),[]));    
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

feat = imfilter(feat,fspecial('gaussian',[9 9],0.5*pixel_step),'same','replicate');
% feat = cv.GaussianBlur_t(feat,'KSize',[ksize ksize],'SigmaX',0.5*pixel_step);

