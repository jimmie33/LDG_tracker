function likelihood=getLike(r1,r2)

% cov_a=cov(a.Feat(chs,:)');
% cov_b=cov(b.Feat(chs,:)');
% 
%for numeric stability

% [U_a,S_a,V_a]=svd(r1);
% S_a=diag(diag(S_a)+0.0000001./(diag(S_a)+0.000001));
% r1 = U_a*(S_a)*U_a';
% 
% [U_b,S_b,V_b]=svd(r2);
% S_b=diag(diag(S_b)+0.0000001./(diag(S_b)+0.000001));
% r2 = U_b*(S_b)*U_b';

e=eig(r1,r2);

s=sqrt(sum(log(abs(e)).^2));

likelihood = exp(-s); 