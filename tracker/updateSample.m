function updateSample(I_vf,sample_sz,radius)
global sampler;
global svm_tracker;

roi_reg = sampler.roi; roi_reg(3:4) = sampler.roi(3:4)-sampler.roi(1:2);
refer_win = svm_tracker.output;
refer_win(1:2) = 0.5*(roi_reg(3:4)-refer_win(3:4));

if refer_win(1)< 1 || refer_win(2) < 1 || refer_win(1) +refer_win(3) > size(I_vf,2) ||...
        refer_win(2)+refer_win(4) > size(I_vf,1)
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

% sampler.patterns_dt=zeros(size(sampler.state_dt,1),size(sampler.template,1)*size(sampler.template,2)*size(I_vf,3));
max_confidence = -inf;
for i=1:size(sampler.state_dt,1)
    rect=temp(i,:);
    sub_win=I_vf(round(rect(2): sampler.step:(rect(2)+rect(4))),round(rect(1): sampler.step : (rect(1)+rect(3))),:);
    confidence = -(sub_win(:)'*svm_tracker.w'+svm_tracker.Bias);
    if confidence > max_confidence
        max_confidence = confidence;
        svm_tracker.output = sampler.state_dt(i,:);
        svm_tracker.confidence = confidence;
    end
%     sampler.patterns_dt(i,:) = sub_win(:)';
end

%% exclude invalide samples
% sampler.patterns_dt = sampler.patterns_dt(valid_sample,:);


end

