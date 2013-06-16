function feature_img = IIF2(img, hist_mtx, k, nbin)
    
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

    %p_sum = sum(hist_mtx, 3);
    up_sum = sum(hist_mtx.*(bp_mtx > b_mtx), 3);
    feature_img = up_sum;%abs(up_sum./p_sum-0.5);  
    
end