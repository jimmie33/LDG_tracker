function updateSample(I_vf)
global sampler;
global svm_tracker;

nel=size(sampler.state,1);
sampler.patterns_dt=zeros(nel,size(sampler.template,1)*size(sampler.template,2)*size(I_vf,3));

valid_sample = boolean(zeros(1,nel));

for i=round(1:nel)
   rect=sampler.state(i,:);
   upleft = round([rect(1)-sampler.roi(1)+1,rect(2)-sampler.roi(2)+1]);
   if ~((upleft(1)<1) || (upleft(2)<1) || (upleft(1)+rect(3)>size(I_vf,2)) || (upleft(2)+rect(4)>size(I_vf,1)))
       sub_win=I_vf(round(upleft(2): sampler.step:(upleft(2)+rect(4))),round(upleft(1): sampler.step : (upleft(1)+rect(3))),:);
      
       sampler.patterns_dt(i,:) = sub_win(:)';
       valid_sample(i) = 1;%valid sample
   end
end


%% compute cost table
left = max(round(sampler.state(:,1)),round(svm_tracker.output(1)));
top = max(round(sampler.state(:,2)),round(svm_tracker.output(2)));
right = min(round(sampler.state(:,1)+sampler.state(:,3)),round(svm_tracker.output(1)+svm_tracker.output(3)));
bottom = min(round(sampler.state(:,2)+sampler.state(:,4)),round(svm_tracker.output(2)+svm_tracker.output(4)));
ovlp = max(right - left,0).*max(bottom - top, 0);
sampler.costs = 1 - ovlp./(2*svm_tracker.output(3)*svm_tracker.output(4)-ovlp);
sampler.state_dt = sampler.state;

%% exclude invalide samples
sampler.costs = sampler.costs(valid_sample);
sampler.patterns_dt = sampler.patterns_dt(valid_sample,:);
sampler.state_dt = sampler.state_dt(valid_sample,:);

end

