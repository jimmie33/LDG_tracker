function updateTracker(I_vf)
global tracker;

nel=size(tracker.state,1);

% tracker.patterns_ft=zeros(nel,size(I_vf,3)); % training sample for feature selection
% tracker.patterns_sp=zeros(nel,tracker.rn*tracker.cn);
tracker.patterns_dt=zeros(nel,size(tracker.template,1)*size(tracker.template,2)*size(I_vf,3));

weight=zeros(1,nel);
valid_sample = boolean(zeros(1,nel));

% [ft_min, ~] = min(tracker.ft_w); % weights should be negative
% ft_w_exp = kron(tracker.ft_w==ft_min,ones(1,size(I_vf,3)/tracker.feature_num));
% tracker.feat_w = ft_w_exp;
med_score = 10*ones(size(tracker.state,1),1);

for i=round(1:nel)
   rect=tracker.state(i,:);
   upleft = round([rect(1)-tracker.roi(1)+1,rect(2)-tracker.roi(2)+1]);
   if ~((upleft(1)<1) || (upleft(2)<1) || (upleft(1)+rect(3)>size(I_vf,2)) || (upleft(2)+rect(4)>size(I_vf,1)))
       sub_win=I_vf(round(upleft(2): tracker.step:(upleft(2)+rect(4))),round(upleft(1): tracker.step : (upleft(1)+rect(3))),:);
       %sub_win=reshape(sub_win,size(sub_win,1)*size(sub_win,2),size(sub_win,3));
       %r1=cov(sub_win);
       %weight(i)=getLike(r1,tracker.template);
       %%
       diff = (sub_win - tracker.template);
      
       tracker.patterns_dt(i,:) = sub_win(:)';
       valid_sample(i) = 1;%valid sample

%        wt_diff = diff(:,:,ft_w_exp>0);%diff.*ft_w_exp;
%        wt_diff = mean(wt_diff,3);
       
       med_score(i) = norm(diff(:))/sqrt(size(diff(:),1));
       weight(i) = exp(-100*med_score(i));%exp(-10*avg_diff);
   end
end

%% test
% [sample, sample_mask] = getSample(I_vf,tracker);
% diff_2d = abs(sample - repmat(tracker.template(:),[1,size(sample,2)]));
% diff_2d = diff_2d(:,sample_mask);
% diff_3d = reshape(diff_2d,size(tracker.template,1)*size(tracker.template,2),size(tracker.template,3),[]);
% tracker.patterns_ft(sample_mask,:)=reshape(mean(diff_3d,1),size(tracker.template,3),[])';
% diff_4d = reshape(diff_2d,size(tracker.template,1),size(tracker.template,2),size(tracker.template,3),[]);
% med_1d = median( reshape(mean(diff_4d(:,:,ft_w_exp>0,:),3),size(tracker.template,1)*size(tracker.template,2),[]), 1);
% weight(sample_mask) = exp(-100*med_1d);
% valid_sample = sample_mask;


%% 
%weight
% w_sort=sort(weight,'descend');
% w_sort(1:10);
[a idx]=max(weight);
tracker.med_score = min(med_score);
%%
tracker.output=tracker.state(idx,:);

%% compute cost table
left = max(round(tracker.state(:,1)),round(tracker.output(1)));
top = max(round(tracker.state(:,2)),round(tracker.output(2)));
right = min(round(tracker.state(:,1)+tracker.state(:,3)),round(tracker.output(1)+tracker.output(3)));
bottom = min(round(tracker.state(:,2)+tracker.state(:,4)),round(tracker.output(2)+tracker.output(4)));
ovlp = max(right - left,0).*max(bottom - top, 0);
tracker.costs = 1 - ovlp./(2*tracker.output(3)*tracker.output(4)-ovlp);

% tracker.output_code = bi2de(((samples(idx,:)-0.5)*tracker.hash_mtx)>0);
% 
% if size(tracker.pos_codes,1)>0
%     tracker.pos_hist = tracker.pos_hist + hist(bi2de(tracker.pos_codes),0:size(tracker.pos_hist,2)-1);
% end
% if size(tracker.neg_codes,1)>0
%     tracker.neg_hist = tracker.neg_hist + hist(bi2de(tracker.neg_codes),0:size(tracker.neg_hist,2)-1);
% end

tracker.state_dt = tracker.state;

% swap the position
idxls = 1:nel; 
valid_idx = idxls(valid_sample); 

temp = tracker.costs(idx);
tracker.costs(idx) = tracker.costs(valid_idx(1));
tracker.costs(valid_idx(1)) = temp;

temp = tracker.patterns_dt(idx,:);
tracker.patterns_dt(idx,:) = tracker.patterns_dt(valid_idx(1),:);
tracker.patterns_dt(valid_idx(1),:) = temp;

temp = tracker.state_dt(idx,:);
tracker.state_dt(idx,:) = tracker.state_dt(valid_idx(1),:);
tracker.state_dt(valid_idx(1),:) = temp;


% exclude invalide samples

tracker.costs = tracker.costs(valid_sample);
tracker.patterns_dt = tracker.patterns_dt(valid_sample,:);
tracker.state_dt = tracker.state_dt(valid_sample,:);
%%
% init_rect = tracker.output;
% tracker.shrange = [init_rect(1)-init_rect(3),init_rect(2)-init_rect(4),...
%     init_rect(1)+2*init_rect(3), init_rect(2)+2*init_rect(4)];
% tracker.topK=tracker.state((find(weight>=w_sort(10))),:);

%% resampling
weight=round(nel*cumsum(weight)/sum(weight));

temp=zeros(size(tracker.state));
counter=1;
for i=1:nel
    s=counter;
    for j=s:weight(i)
        temp(counter,1:2)=tracker.state(i,1:2)+1*(rand(1,2)-0.5)*(tracker.radius*2*max(tracker.state(i,3:4)));
        temp(counter,3:4)=tracker.state(i,3:4);
        counter=counter+1;
        if counter>nel
            break;
        end
    end
    if counter>nel
        break;
    end
end
tracker.state=temp;

%% update template
% rect = tracker.output;
% rect(1:2) = rect(1:2)-tracker.roi(1:2)+1;
% sub_win=I_vf(round(rect(2): tracker.step:(rect(2)+rect(4))),round(rect(1): tracker.step : (rect(1)+rect(3))),:);
% %diff = sum(abs(sub_win - tracker.template),3);
% %diff_md = median(diff(:));
% %mask = (diff > 0.96*diff_md)&(diff < 1.04*diff_md);
% %update_mask = repmat(mask,[1,1,size(tracker.template,3)]);
% % update_rate = repmat(reshape(tracker.ft_w,1,1,[]),[size(tracker.template,1),size(tracker.template,2)]);
% % update_rate = exp(-30*max(update_rate,0)./sum(tracker.ft_w(:)));
% update_vector = sub_win-tracker.template;
% tracker.template = tracker.template + tracker.ln_rate*sign(update_vector).*abs(update_vector);
% % sprintf('max: %f, min: %f',max(update_rate(:)),min(update_rate(:)))
% display(size(sub_win,1)*size(sub_win,2))
%display(sum(mask(:)))

end

