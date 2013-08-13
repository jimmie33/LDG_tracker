function updateSample(I_vf,I,sample_sz,radius)
global sampler;
global svm_tracker;

roi_reg = sampler.roi; roi_reg(3:4) = sampler.roi(3:4)-sampler.roi(1:2);
refer_win = svm_tracker.output;
refer_win(1:2) = 0.5*(roi_reg(3:4)-refer_win(3:4));

if refer_win(1)< 1 || refer_win(2) < 1 || refer_win(1) +refer_win(3)-1 > size(I_vf,2) ||...
        refer_win(2)+refer_win(4)-1 > size(I_vf,1)
    error('out of border')
end

temp = repmatls(refer_win,[sample_sz,1]);
rad = (rand(size(temp,1)-1,1))*(radius*max([sampler.template_width,sampler.template_height]));
angle = rand(size(temp,1)-1,1)*2*pi;
temp(2:end,1:2) = temp(2:end,1:2) + [cos(angle).*rad,sin(angle).*rad];


% valid_sample = boolean(zeros(1,sample_sz));
valid_sample = ~(temp(:,1)<1 | temp(:,2)<1 | temp(:,1)+temp(:,3)>size(I_vf,2) | temp(:,2)+temp(:,4)>size(I_vf,1));
temp = temp(valid_sample,:);
sampler.state_dt = temp;
sampler.state_dt(:,1) = sampler.state_dt(:,1)+sampler.roi(1)-1;
sampler.state_dt(:,2) = sampler.state_dt(:,2)+sampler.roi(2)-1;

% max_confidence = -inf;
% max_count = -inf;
confidence = zeros(size(sampler.state_dt,1),1);
if svm_tracker.state == 1
    confidence_old = zeros(size(sampler.state_dt,1),1);
end
for i=1:size(sampler.state_dt,1)
    rect = temp(i,:);
    sub_win = I_vf(round(rect(2): sampler.step:(rect(2)+rect(4)-1)),round(rect(1): sampler.step : (rect(1)+rect(3)-1)),:);
    sub_win = sub_win(:)';
    confidence(i) = -(sub_win*svm_tracker.w'+svm_tracker.Bias);
    if svm_tracker.state == 1
        confidence_old(i) = -(sub_win*svm_tracker.snapshot.w'+svm_tracker.snapshot.Bias);
    end
%     sub_win_raw = I(round(rect(2): sampler.step:(rect(2)+rect(4))),round(rect(1): sampler.step : (rect(1)+rect(3))),:);
%     sub_win_raw = sub_win_raw(:)';
%     hs_code = bi2de(sub_win_raw(sampler.hash_function.a) > sampler.hash_function.b)+1;
%     hs_idx = sub2ind(size(sampler.hash_table),[1:8]',hs_code);
%     count = sum(sampler.hash_table(hs_idx));
%     if confidence > max_confidence
%         max_confidence = confidence;
%         svm_tracker.output = sampler.state_dt(i,:);
%         svm_tracker.confidence = confidence;
%         svm_tracker.output_feat = sub_win;
%         svm_tracker.output_feat_raw = sub_win_raw;
%     end
%     if count > max_count
%         max_count = count;
%         svm_tracker.output_debug = sampler.state_dt(i,:);
%     end
%     sampler.patterns_dt(i,:) = sub_win(:)';
end

[Y Id] = max(confidence);

svm_tracker.output = sampler.state_dt(Id,:);
svm_tracker.confidence = confidence(Id);
rect = temp(Id,:);
sub_win = I_vf(round(rect(2): sampler.step:(rect(2)+rect(4)-1)),round(rect(1): sampler.step : (rect(1)+rect(3)-1)),:);
sub_win_raw = I(round(rect(2): sampler.step:(rect(2)+rect(4)-1)),round(rect(1): sampler.step : (rect(1)+rect(3)-1)),:);
svm_tracker.output_feat = sub_win(:)';
svm_tracker.output_feat_raw = sub_win_raw(:)';

%compute ambiguity loss
iou = getIOU(sampler.state_dt,svm_tracker.output);
temp_mask = iou <0.1;
temp_lab = [-1*ones(sum(temp_mask),1)];
temp_conf = [confidence(temp_mask)];
temp_gain = temp_conf.*temp_lab;
[val temp_idx]=min(temp_gain);
amb_idx = temp_gain < 1; amb_idx(temp_idx) = true;
temp_lab_alt = temp_lab; temp_lab_alt(amb_idx) = -temp_lab_alt(amb_idx);
temp_diff = 2*min([sum(amb_idx)]);
svm_tracker.ambiguity_loss = max(temp_conf'*(temp_lab_alt - temp_lab)+temp_diff,0);

if svm_tracker.state == 1
    [Y_old Id_old] = max(confidence_old);

    svm_tracker.output_old = sampler.state_dt(Id_old,:);
    svm_tracker.confidence_old = confidence_old(Id_old);
    rect = temp(Id_old,:);
    sub_win = I_vf(round(rect(2): sampler.step:(rect(2)+rect(4)-1)),round(rect(1): sampler.step : (rect(1)+rect(3)-1)),:);
    sub_win_raw = I(round(rect(2): sampler.step:(rect(2)+rect(4)-1)),round(rect(1): sampler.step : (rect(1)+rect(3)-1)),:);
    svm_tracker.output_feat_old = sub_win(:)';
    svm_tracker.output_feat_raw_old = sub_win_raw(:)';
    
    svm_tracker.confidence_old_new = -(svm_tracker.output_feat_old*svm_tracker.w'+svm_tracker.Bias);
    
    % compute ambiguity loss
    
 
    iou_old = getIOU(sampler.state_dt,svm_tracker.output_old);
    temp_mask_old = iou_old<0.1;  
    temp_lab_old = [-1*ones(sum(temp_mask_old),1)];
    temp_conf_old = [confidence_old(temp_mask_old)];
 
    temp_gain_old = temp_conf_old.*temp_lab_old;
    [val temp_idx_old]=min(temp_gain_old);
    amb_idx_old = temp_gain_old < 1; amb_idx_old(temp_idx_old) = true;

    temp_lab_alt_old = temp_lab_old; temp_lab_alt_old(amb_idx_old) = -temp_lab_alt_old(amb_idx_old);
    temp_diff_old = 2*min([sum(amb_idx_old)]);
    svm_tracker.ambiguity_loss_old = max(temp_conf_old'*(temp_lab_alt_old - temp_lab_old)+temp_diff_old,0);        
end

% svm_tracker.output_feat_raw = sub_win_raw(:)';

% iou = getIOU(sampler.state_dt,repmatls(svm_tracker.output,[size(sampler.state_dt,1),1]));
% [aa iidx]=max(iou(Id)<0.5);
% svm_tracker.output_second = sampler.state_dt(Id(iidx),:);


%% exclude invalide samples
% sampler.patterns_dt = sampler.patterns_dt(valid_sample,:);


end

