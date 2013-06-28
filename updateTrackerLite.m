function updateTrackerLite(I_vf)
global tracker;
nel=size(tracker.state,1);
tracker.patterns_dt=zeros(nel,size(tracker.template,1)*size(tracker.template,2)*size(I_vf,3));
valid_sample = boolean(zeros(1,nel));

for i=round(1:nel)
   rect=tracker.state(i,:);
   upleft = round([rect(1)-tracker.roi(1)+1,rect(2)-tracker.roi(2)+1]);
   if ~((upleft(1)<1) || (upleft(2)<1) || (upleft(1)+rect(3)>size(I_vf,2)) || (upleft(2)+rect(4)>size(I_vf,1)))
       sub_win=I_vf(round(upleft(2): tracker.step:(upleft(2)+rect(4))),round(upleft(1): tracker.step : (upleft(1)+rect(3))),:);
      
       tracker.patterns_dt(i,:) = sub_win(:)';
       valid_sample(i) = 1;%valid sample
   end
end

tracker.state_dt = tracker.state(valid_sample,:);
tracker.patterns_dt = tracker.patterns_dt(valid_sample,:);

end