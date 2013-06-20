% Demo for paper "Visual Tracking via Locality Sensitive Histograms" 
% by Shengfeng He, Qingxiong Yang, Rynson W.H. Lau, Jiang Wang, and Ming-Hsuan Yang
% To appear in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2013), Portland, June, 2013.
% feature_img output a (x,y) matrix.
function feature_img = IIF(img, hist_mtx, k, nbin)
    
    color_max = 256;
    color_range = 0:color_max/nbin:color_max;
    % find bin id for each pixel
    i_mtx = repmatls(img,[1,1,nbin]);
    l = repmatls(reshape(color_range(1:end-1),1,1,[]),[size(img,1),size(img,2)]);
    u = repmatls(reshape(color_range(2:end),1,1,[]),[size(img,1),size(img,2)]);
    b_mtx = repmatls(reshape(1:nbin,1,1,[]),[size(img,1),size(img,2),1]);
    
    mask = (i_mtx>=l & i_mtx<u);
    bp_mtx = zeros(size(img, 1), size(img, 2),nbin);
    bp_mtx(mask) = b_mtx(mask);
    bp_mtx = sum(bp_mtx,3);
    bp_mtx = repmatls(bp_mtx, [1,1, nbin]);
    
    i_mtx = k * i_mtx;
    i_mtx(i_mtx < k) = k;

    % compute illumination invariant features
    X = -max((b_mtx - bp_mtx),0).^2 ./ (2 * i_mtx.^2);
    e_mtx = exp(X);
    Ip_mtx = e_mtx .* hist_mtx;
    feature_img = sum(Ip_mtx, 3);  
    
end