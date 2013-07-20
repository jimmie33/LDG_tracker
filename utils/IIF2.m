function feature_img = IIF2(img, hist_mtx,nbin)
    
    color_max = 256;
    color_range = 0:color_max/nbin:color_max;
    % find bin id for each pixel
    img_rep = repmatls(img,[1,1,nbin]);
    l = repmatls(reshape(color_range(1:end-1),1,1,[]),[size(img,1),size(img,2)]);
    u = repmatls(reshape(color_range(2:end),1,1,[]),[size(img,1),size(img,2)]);
    b_mtx = repmatls(reshape(1:nbin,1,1,[]),[size(img,1),size(img,2),1]);
    
    mask = (img_rep>=l & img_rep<u);
    bp_mtx = zeros(size(img, 1), size(img, 2),nbin);
    bp_mtx(mask) = b_mtx(mask);
    bp_mtx = sum(bp_mtx,3);
    bp_mtx = repmatls(bp_mtx, [1,1, nbin]);

    up_sum = sum(hist_mtx.*(bp_mtx > b_mtx), 3);
    feature_img = up_sum;%abs(up_sum./p_sum-0.5);  
    
end