function updateSample(I_vf,I,sample_sz,radius)
global sampler;
global svm_tracker;

roi_reg = sampler.roi; roi_reg(3:4) = sampler.roi(3:4)-sampler.roi(1:2);
refer_win = svm_tracker.output;
refer_win(1:2) = refer_win(1:2)-roi_reg(1:2)+1;

% if refer_win(1)< 1 || refer_win(2) < 1 || refer_win(1) +refer_win(3)-1 > size(I_vf,2) ||...
%         refer_win(2)+refer_win(4)-1 > size(I_vf,1)
%     keyboard
%     error('out of border')
% end

r = radius*(norm([sampler.template_width,sampler.template_height]))/2;
% x_s = refer_win(1) - r;
% y_s = refer_win(2) - r;
% x_e = refer_win(1) + r;
% y_e = refer_win(2) + r;

step = round(2*r/sqrt(sample_sz));

x_sample = -r:step:r;
y_sample = -r:step:r;
[X Y] = meshgrid(x_sample,y_sample);
% rad_sample_num = round(sample_sz/8);
% rad_step = round(r/rad_sample_num);
% rad_sample = rad_step:rad_step:r;
% round(sqrt(sample_sz)/2);
temp = repmatls(refer_win,[numel(X),1]);

% ang_sample = (0:7)*pi/4;
% ang_sample = repmatls([ang_sample', ang_sample'+ pi/8],[1,ceil(numel(rad_sample)/2)]);
% ang_sample = ang_sample(:,1:numel(rad_sample));
% 
% rad_sample = repmatls(rad_sample,[8,1]);

% rad = (rand(size(temp,1)-1,1))*(radius*max([sampler.template_width,sampler.template_height]));
% angle = rand(size(temp,1)-1,1)*2*pi;
temp(:,1:2) = temp(:,1:2) + [X(:),Y(:)];


% valid_sample = boolean(zeros(1,sample_sz));
valid_sample = ~(temp(:,1)<1 | temp(:,2)<1 | temp(:,1)+temp(:,3)>size(I_vf,2) | temp(:,2)+temp(:,4)>size(I_vf,1));
temp = temp(valid_sample,:);
sampler.state_dt = temp;
sampler.state_dt(:,1) = sampler.state_dt(:,1)+sampler.roi(1)-1;
sampler.state_dt(:,2) = sampler.state_dt(:,2)+sampler.roi(2)-1;

% max_confidence = -inf;
% max_count = -inf;
confidence = zeros(size(sampler.state_dt,1),1);
confidence_exp = zeros(size(sampler.state_dt,1),1);
if svm_tracker.state == 1
    confidence_old = zeros(size(sampler.state_dt,1),1);
end
idx = svm_tracker.best_expert_idx;
expert_w = svm_tracker.experts{idx}.w;
expert_b = svm_tracker.experts{idx}.Bias;
for i=1:size(sampler.state_dt,1)
    rect = temp(i,:);
    sub_win = I_vf(round(rect(2): sampler.step:(rect(2)+rect(4)-1)),round(rect(1): sampler.step : (rect(1)+rect(3)-1)),:);
    sub_win = sub_win(:)';
    confidence_exp(i) = -(sub_win*expert_w'+ expert_b);
    confidence(i) = -(sub_win*svm_tracker.w' + svm_tracker.Bias);
end

[Y Id] = max(confidence);

svm_tracker.output = sampler.state_dt(Id,:);
svm_tracker.confidence = confidence(Id);
rect = temp(Id,:);
sub_win = I_vf(round(rect(2): sampler.step:(rect(2)+rect(4)-1)),round(rect(1): sampler.step : (rect(1)+rect(3)-1)),:);
sub_win_raw = I(round(rect(2): sampler.step:(rect(2)+rect(4)-1)),round(rect(1): sampler.step : (rect(1)+rect(3)-1)),:);
svm_tracker.output_feat = sub_win(:)';
svm_tracker.output_feat_raw = sub_win_raw(:)';

[Y Id] = max(confidence_exp);
svm_tracker.output_exp = sampler.state_dt(Id,:);
svm_tracker.confidence_exp = confidence_exp(Id);

end

