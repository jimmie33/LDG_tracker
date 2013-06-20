function [inv_cov,mn] = getWhitenSpace (I_vf,ft_selected,rn,cn,sample_step,pix_step)
x = 1:sample_step:size(I_vf,2)-cn+1;
y = 1:sample_step:size(I_vf,1)-rn+1;
x_ = repmat(x,[1,size(y,2)]);
y_ = kron(y,ones(1,size(x,2)));
indexFcn = @(y_,x_) I_vf(y_:pix_step:(y_+rn-1),x_:pix_step:(x_+cn-1),ft_selected);
result = arrayfun(indexFcn,y_,x_,'UniformOutput',false);
result = reshape(cat(3,result{:}),(floor((rn-1)/pix_step)+1)*(floor((cn-1)/pix_step)+1)*sum(ft_selected),[]);
mn = mean(result,2);
centered = result - repmat(mn,[1 size(result,2)]);
[U S V] = svd(centered,0);
s_sqr = diag(S).^2;
idx = find(cumsum(s_sqr)>0.99*sum(s_sqr));
inv_cov = U(:,1:idx(1))*diag(1./s_sqr(1:idx(1)))*U(:,1:idx(1))';


