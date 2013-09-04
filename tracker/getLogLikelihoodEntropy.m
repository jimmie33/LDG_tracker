function [ll, entropy] = getLogLikelihoodEntropy (svm_score,label_prior,label_prior_neg)
% compute log(P(theta|L)) = log(P(L|,theta)) + lambda*H(Y|X,Z;L,theta)
% P(y|x,theta) = min(max(0.5*y*svm_score+0.5,0),1)

num = numel(svm_score);
% label_mat = -ones(num,num) + 2*eye(num);
% label_mat = [label_mat];

%     pos_score = min( max((0.5*svm_score+0.5),0+0.001), 1-0.001).^2;
% pos_score = (atan(svm_score)/(pi/2)+1)/2;
pos_score = normcdf(svm_score,0,1);
pos_score = pos_score.*label_prior(:)';
% pos_score = (exp(1*svm_score)./(1+exp(1*svm_score)));
% label_prior = label_prior/sum(label_prior(:));
% display(svm_score')

neg_score = 1 - pos_score;
neg_score = neg_score.*label_prior_neg(:)';
% p_XY_Z = pos_score.*label_prior(:)';
p_XY_Z = prod(repmat(neg_score(:),[1 num])+diag(pos_score - neg_score));
% p_XY_Z = (min( max( 0.5*bsxfun(@times,-ones(num,num),svm_score(:))+0.5, 0), 1));% could threshold it with eps > 0
g_XY_Z = p_XY_Z/sum(p_XY_Z);
entropy = -g_XY_Z*log(g_XY_Z)';% in case g is 0
% label_prior = label_prior/sum(label_prior);
% entropy = -2*g_XY_Z*log(label_prior(:)) - 1*g_XY_Z*log(g_XY_Z)';%%% important
% [val idx] = max(p_XY_Z(:));
% ll = log(val*label_prior(idx)+eps);
ll = log(max(p_XY_Z));
