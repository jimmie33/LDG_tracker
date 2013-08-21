function I_vf = exhausiveSearch(I, scales)

global svm_tracker
global sampler
global config


% scales = svm_tracker.scale*scales;
scales = scales(svm_tracker.scale*scales > svm_tracker.scale_lowbound & svm_tracker.scale*scales < svm_tracker.scale_upbound);
feat_col = {};
sliding_win = {};
best_score = [];
best_score_idx = [];
BC = {};
for i = 1:numel(scales)
    I_scale = imresize(I,1/scales(i));
    BC{i} = getFeatureRep(I_scale,config.hist_nbin,config.pixel_step);
    
    sliding_win{i} = struct();
    sliding_win{i}.step_size = max(round(sampler.template_size(1:2)/5),1);
    feature_map = imresize(BC{i},1/config.pixel_step,'nearest');
    feat_col{i} = im2colstep(feature_map,sampler.template_size,...
    [sliding_win{i}.step_size, sampler.template_size(3)]);
    sliding_win{i}.patch_size = sampler.template_size;
    sliding_win{i}.map_size = size(feature_map);
    map_svm = getConfidenceMap(feat_col{i},-svm_tracker.w,-svm_tracker.Bias,sliding_win{i});
    [best_score(i) best_score_idx(i)] = max(map_svm(:));
end

[score,idx] = max(best_score);
I_vf = BC{idx};
% map_svm = getConfidenceMap(feat_col{idx},-svm_tracker.snapshot.w,-svm_tracker.snapshot.Bias,sliding_win{idx});
% map_svm = -svm_tracker.snapshot.w*feat_col{idx}-svm_tracker.Bias;
% score = map_svm(best_score_idx(idx));
if score > 0.01
    map_svm = getConfidenceMap(feat_col{idx},-svm_tracker.w,-svm_tracker.Bias,sliding_win{idx});
    map_diff = -getDifferenceMap(feat_col{idx},sampler.template',sliding_win{idx});
    
    map_svm = (map_svm-min(map_svm(:)))/(max(map_svm(:))-min(map_svm(:)));
    map_diff = (map_diff-min(map_diff(:)))/(max(map_diff(:))-min(map_diff(:)));
% %     
%     figure(4)
%     subplot(1,2,1)
%     imagesc(map_svm)
%     subplot(1,2,2)
%     imagesc(map_diff)
%     
    map_agree = (map_svm > 0.95) + (map_diff >0.95);
    [agree_score, agree_idx] = max(map_agree(:));
    if agree_score > 1
        config.tracking_failure = false;
        svm_tracker.scale = scales(idx)*svm_tracker.scale;
        svm_tracker.confidence = score;
        sampler.roi = sampler.roi/scales(idx);
        rect = svm_tracker.output;
        [r c] = ind2sub(size(map_svm),best_score_idx(idx));
        rect(1:2) = ([c r]-1).*sliding_win{i}.step_size([2 1]) + 1;
        rect(1:2) = ((rect(1:2)-1)*config.pixel_step + sampler.roi(1:2));
        svm_tracker.output = rect;
    end
end