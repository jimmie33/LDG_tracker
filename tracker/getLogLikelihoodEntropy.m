function [ll, entropy] = getLogLikelihoodEntropy (svm_score,is_svm)
% compute log(P(theta|L)) = log(P(L|,theta)) + lambda*H(Y|X,Z;L,theta)
% P(y|x,theta) = min(max(0.5*y*svm_score+0.5,0),1)

num = numel(svm_score);
% label_mat = -ones(num,num) + 2*eye(num);
% label_mat = [label_mat];
if is_svm
%     pos_score = min( max((0.5*svm_score+0.5),0+0.001), 1-0.001).^2;
    pos_score = (atan(1*svm_score)/(pi/2)+1)/2;
else
    pos_score = svm_score;
end
neg_score = 1 - pos_score;
p_XY_Z = prod(repmatls(neg_score(:),[1 num])+diag(pos_score - neg_score));
% p_XY_Z = (min( max( 0.5*bsxfun(@times,-ones(num,num),svm_score(:))+0.5, 0), 1));% could threshold it with eps > 0
g_XY_Z = p_XY_Z/sum(p_XY_Z);
entropy = -g_XY_Z*log(g_XY_Z)';% in case g is 0
ll = log(sum(p_XY_Z)+eps);
