% Demo for paper "Visual Tracking via Locality Sensitive Histograms" 
% by Shengfeng He, Qingxiong Yang, Rynson W.H. Lau, Jiang Wang, and Ming-Hsuan Yang
% To appear in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2013), Portland, June, 2013.
% hist_mtx output a (x,y,b) matrix
function hist_mtx = LSH(img, alpha, nbin)

    color_max = 256;
    color_range = 0:color_max/nbin:color_max;
    
    img_rep = repmat(img,[1,1,nbin]);
    l = repmat(reshape(color_range(1:end-1),1,1,[]),[size(img,1),size(img,2)]);
    u = repmat(reshape(color_range(2:end),1,1,[]),[size(img,1),size(img,2)]);
    q_mtx = double((img_rep >= l) & (img_rep<u));

%     % compute Q
%     q_mtx = zeros(size(img, 1), size(img, 2), nbin);
%     for i=1:nbin
%         tmp_img = img;
% 
%         mask_l = find(tmp_img >= color_range(i));
%         mask_u = find(tmp_img < color_range(i+1));
%         mask = intersect(mask_l, mask_u);
% 
%         tmp_img(:) = 0;
%         tmp_img(mask) = 1;
%         q_mtx(:, :, i) = tmp_img;
%     end

    % compute H and normalization factor
%     hist_mtx = q_mtx;
    
    kernel = getExpKernel1D(alpha,2*round( log(0.5*0.025*(1-alpha))/log(alpha)-1 )+1);
    hist_mtx = imfilter(q_mtx,kernel,'same','replicate');
    hist_mtx = imfilter(hist_mtx,kernel','same','replicate');

end

%%
function kernel = getExpKernel1D(alpha,sz)
    indices = [ceil(sz/2):-1:1,2:ceil(sz/2)];
    kernel = alpha.^indices;
    kernel = kernel/sum(kernel);
end
