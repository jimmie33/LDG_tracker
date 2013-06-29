function resample(I_vf,sample_sz,radius)
global sampler;
global svm_tracker;

%% updata sampler states

temp = repmatls(svm_tracker.output,[sample_sz,1]);
rad = (rand(size(temp,1)-1,1))*(radius*max([sampler.template_width,sampler.template_height]));
angle = rand(size(temp,1)-1,1)*2*pi;
temp(2:end,1:2) = temp(2:end,1:2) + [cos(angle).*rad,sin(angle).*rad];

%% update training samples
sampler.patterns_dt=zeros(sample_sz,size(sampler.template,1)*size(sampler.template,2)*size(I_vf,3));
valid_sample = boolean(zeros(1,sample_sz));
for i=1:sample_sz
   rect=temp(i,:);
   upleft = round([rect(1)-sampler.roi(1)+1,rect(2)-sampler.roi(2)+1]);
   if ~((upleft(1)<1) || (upleft(2)<1) || (upleft(1)+rect(3)>size(I_vf,2)) || (upleft(2)+rect(4)>size(I_vf,1)))
       sub_win=I_vf(round(upleft(2): sampler.step:(upleft(2)+rect(4))),round(upleft(1): sampler.step : (upleft(1)+rect(3))),:);
       sampler.patterns_dt(i,:) = sub_win(:)';
       valid_sample(i) = 1;
   end
end

%% compute cost table
left = max(round(temp(:,1)),round(svm_tracker.output(1)));
top = max(round(temp(:,2)),round(svm_tracker.output(2)));
right = min(round(temp(:,1)+temp(:,3)),round(svm_tracker.output(1)+svm_tracker.output(3)));
bottom = min(round(temp(:,2)+temp(:,4)),round(svm_tracker.output(2)+svm_tracker.output(4)));
ovlp = max(right - left,0).*max(bottom - top, 0);
sampler.costs = 1 - ovlp./(2*svm_tracker.output(3)*svm_tracker.output(4)-ovlp);

sampler.costs = sampler.costs(valid_sample);
sampler.patterns_dt = sampler.patterns_dt(valid_sample,:);
sampler.state_dt = temp(valid_sample,:);

end

