function tracker=correctTracker(tracker,I_vf,idx)


%% updata tracker states
tracker.output=tracker.state_dt(idx,:);
temp = repmatls(tracker.output,[round(size(tracker.state,1)/1),1]);
rad = (rand(size(temp,1),1))*(1.3*max(tracker.state(1,3:4)));
angle = rand(size(temp,1),1)*2*pi;
temp(:,1:2) = temp(:,1:2) + [cos(angle).*rad,sin(angle).*rad];
temp(1,:) = tracker.output;% at least one original gt
tracker.state(randsample(size(tracker.state,1),size(temp,1)),:) = temp;
temp = tracker.state;

%% update training samples
nel = size(temp,1);
tracker.patterns_dt=zeros(nel,size(tracker.template,1)*size(tracker.template,2)*size(I_vf,3));
valid_sample = boolean(zeros(1,nel));
for i=round(1:nel)
   rect=temp(i,:);
   upleft = round([rect(1)-tracker.roi(1)+1,rect(2)-tracker.roi(2)+1]);
   if ~((upleft(1)<1) || (upleft(2)<1) || (upleft(1)+rect(3)>size(I_vf,2)) || (upleft(2)+rect(4)>size(I_vf,1)))
       sub_win=I_vf(round(upleft(2): tracker.step:(upleft(2)+rect(4))),round(upleft(1): tracker.step : (upleft(1)+rect(3))),:);
       %%
       tracker.patterns_dt(i,:) = sub_win(:)';
       valid_sample(i) = 1;
   end
end

%% compute cost table
left = max(round(temp(:,1)),round(tracker.output(1)));
top = max(round(temp(:,2)),round(tracker.output(2)));
right = min(round(temp(:,1)+temp(:,3)),round(tracker.output(1)+tracker.output(3)));
bottom = min(round(temp(:,2)+temp(:,4)),round(tracker.output(2)+tracker.output(4)));
ovlp = max(right - left,0).*max(bottom - top, 0);
tracker.costs = 1 - ovlp./(2*tracker.output(3)*tracker.output(4)-ovlp);

tracker.costs = tracker.costs(valid_sample);
tracker.patterns_dt = tracker.patterns_dt(valid_sample,:);
tracker.state_dt = temp(valid_sample,:);

%% update template
rect = tracker.output;
rect(1:2) = rect(1:2)-tracker.roi(1:2)+1;
sub_win=I_vf(round(rect(2): tracker.step:(rect(2)+rect(4))),round(rect(1): tracker.step : (rect(1)+rect(3))),:);

update_vector = sub_win-tracker.template;
tracker.template = tracker.template + tracker.ln_rate*update_vector;

end

