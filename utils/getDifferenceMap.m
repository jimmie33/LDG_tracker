function M = getDifferenceMap(feat_col,template,sliding_win)

dist = (bsxfun(@minus,feat_col,template));
% dist_projected = (w/norm(w))*dist;
dist_norm2 = sum(dist.^2,1);
M = sqrt(dist_norm2);
mp_sz = floor((sliding_win.map_size(1:2) - sliding_win.patch_size(1:2))./sliding_win.step_size(1:2))+1;
M = reshape(M,mp_sz);