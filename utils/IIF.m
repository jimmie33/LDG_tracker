% Demo for paper "Visual Tracking via Locality Sensitive Histograms" 
% by Shengfeng He, Qingxiong Yang, Rynson W.H. Lau, Jiang Wang, and Ming-Hsuan Yang
% To appear in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2013), Portland, June, 2013.
% feature_img output a (x,y) matrix.
function feature_img = IIF(img, hist_mtx, k, nbin)
    
    color_max = 256;
    color_range = 0:color_max/nbin:color_max;
    % find bin id for each pixel
    bp_mtx = zeros(size(img, 1), size(img, 2));
    for i=1:nbin
        mask_l = find(img >= color_range(i));
        mask_u = find(img < color_range(i+1));
        mask = intersect(mask_l, mask_u);
        bp_mtx(mask) = i;
    end
    bp_mtx = repmat(bp_mtx, [1,1, nbin]);

    % construct bin id matrix
    b_mtx(1,1,:) = 1:nbin;
    b_mtx = repmat(b_mtx, [size(img, 1), size(img, 2), 1]);

    % contruct pixel intensity matrix
    i_mtx = repmat(double(img), [1, 1, nbin]);
    
    i_mtx = k * i_mtx;
    i_mtx(i_mtx < k) = k;

    % compute illumination invariant features
    X = -max((b_mtx - bp_mtx),0).^2 ./ (2 * i_mtx.^2);
    e_mtx = exp(X);
    Ip_mtx = e_mtx .* hist_mtx;
    feature_img = sum(Ip_mtx, 3);  
    
end