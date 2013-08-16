function resample(I_vf,step_size)
global sampler;
global svm_tracker;
global config;

if nargin < 2
    step_size = max(round(sampler.template_size(1:2)/4),1);
end

feature_map = imresize(I_vf,1/config.pixel_step,'nearest');
% step_size = max(round(sampler.template_size(1:2)/4),1);

rect=svm_tracker.output_exp;
upleft = round([rect(1)-sampler.roi(1)+1,rect(2)-sampler.roi(2)+1]);
if ~((upleft(1)<1) || (upleft(2)<1) || (upleft(1)+rect(3)-1>size(I_vf,2)) || (upleft(2)+rect(4)-1>size(I_vf,1)))
    sub_win=I_vf(round(upleft(2): sampler.step:(upleft(2)+rect(4)-1)),round(upleft(1): sampler.step : (upleft(1)+rect(3)-1)),:);
    svm_tracker.output_feat = sub_win(:)';
else    
%     sub_win=I_vf(round(upleft(2): sampler.step:(upleft(2)+rect(4))),round(upleft(1): sampler.step : (upleft(1)+rect(3))),:);
%     svm_tracker.output_feat = sub_win(:)';
    warning('tracking window outside of frame');
    keyboard
end

sampler.patterns_dt = [im2colstep(feature_map,sampler.template_size,[step_size, size(I_vf,3)])';...
    svm_tracker.output_feat];
if(config.ellipse_mask)
    sampler.patterns_dt(:,sampler.mask(:)) = 0;
end
temp = repmatls(svm_tracker.output_exp,[size(sampler.patterns_dt,1),1]);

[X Y] = meshgrid(1:step_size(2):size(feature_map,2)-sampler.template_size(2)+1,1:step_size(1):size(feature_map,1)-sampler.template_size(1)+1);
temp(1:end-1,1) = (X(:)-1)*config.pixel_step + sampler.roi(1);
temp(1:end-1,2) = (Y(:)-1)*config.pixel_step + sampler.roi(2);


%% compute cost table
left = max(round(temp(:,1)),round(svm_tracker.output_exp(1)));
top = max(round(temp(:,2)),round(svm_tracker.output_exp(2)));
right = min(round(temp(:,1)+temp(:,3)),round(svm_tracker.output_exp(1)+svm_tracker.output_exp(3)));
bottom = min(round(temp(:,2)+temp(:,4)),round(svm_tracker.output_exp(2)+svm_tracker.output_exp(4)));
ovlp = max(right - left,0).*max(bottom - top, 0);
sampler.costs = 1 - ovlp./(2*svm_tracker.output_exp(3)*svm_tracker.output_exp(4)-ovlp);

% sampler.costs = sampler.costs(valid_sample);
% sampler.patterns_dt = sampler.patterns_dt(valid_sample,:);
sampler.state_dt = temp;



end

