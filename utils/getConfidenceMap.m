function M = getConfidenceMap(feat_col,w,b,sliding_win)

M = w*feat_col+b;
mp_sz = floor((sliding_win.map_size(1:2) - sliding_win.patch_size(1:2))./sliding_win.step_size(1:2))+1;
M = reshape(M,mp_sz);