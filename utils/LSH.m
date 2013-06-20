% Demo for paper "Visual Tracking via Locality Sensitive Histograms" 
% by Shengfeng He, Qingxiong Yang, Rynson W.H. Lau, Jiang Wang, and Ming-Hsuan Yang
% To appear in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2013), Portland, June, 2013.
% hist_mtx output a (x,y,b) matrix
function hist_mtx = LSH(img, alpha, nbin)

    color_max = 256;
    color_range = 0:color_max/nbin:color_max;
    
    img_rep = repmatls(img,[1,1,nbin]);
    l = repmatls(reshape(color_range(1:end-1),1,1,[]),[size(img,1),size(img,2)]);
    u = repmatls(reshape(color_range(2:end),1,1,[]),[size(img,1),size(img,2)]);
    q_mtx = double((img_rep >= l) & (img_rep<u));

    % 95% accuracy kernel
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
