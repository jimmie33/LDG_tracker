function evaluateExperts(I_vf,lambda)
global sampler;
global svm_tracker;
global config;

feature_map = imresize(I_vf,1/config.pixel_step,'nearest');
step_size = max(round(sampler.template_size(1:2)/10),1);

if size(I_vf,3)>1
    patterns_dt = im2colstep(feature_map,sampler.template_size,[step_size, size(I_vf,3)]);
else
    patterns_dt = im2colstep(feature_map,sampler.template_size,step_size);
end

rects = repmatls(svm_tracker.output,[size(patterns_dt,2),1]);

x = 1:step_size(2):size(feature_map,2)-sampler.template_size(2)+1;
y = 1:step_size(1):size(feature_map,1)-sampler.template_size(1)+1;
[X Y] = meshgrid(x,y);
rects(1:end,1) = (X(:)-1)*config.pixel_step + sampler.roi(1);
rects(1:end,2) = (Y(:)-1)*config.pixel_step + sampler.roi(2);
mask_temp = zeros(numel(y),numel(x));
mask_temp(1:5:end,1:5:end) = 1;
% nrow = numel(1:5:numel(y));
% ncol = numel(1:5:numel(x));




%% compute log likelihood and entropy
n = numel(svm_tracker.experts);
score_temp = zeros(n,1);
rect_temp = zeros(n,4);
% figure(4)
% 
% loglik_vec=[];
% ent_vec=[];
% figure(3)

svm_tracker.experts{2}.w = 0.99*svm_tracker.experts{2}.w + 0.01*svm_tracker.experts{3}.w;
svm_tracker.experts{2}.Bias = 0.99*svm_tracker.experts{2}.Bias + 0.01*svm_tracker.experts{3}.Bias;
for i = 1:n

    svm_score = -(svm_tracker.experts{i}.w*patterns_dt+svm_tracker.experts{i}.Bias);
    
    
%     subplot(2,3,i)
% %     svm_score = svm_score + 1 - val;% shift so that the max value is 1
%     
%     imagesc(reshape(svm_score,numel(y),[]));
%     colorbar
%     svm_score = 2*(svm_score - min(svm_score))/(max(svm_score) - min(svm_score))-1;%normalize
    [val idx] = max(svm_score);
    
    best_rect = rects(idx,:);
    rect_temp(i,:) = best_rect;

    [loglik ent] = getLogLikelihoodEntropy([val,svm_score(getIOU(rects,best_rect)<0.5 & mask_temp(:) > 0)],true);
%     loglik_vec(end+1) = loglik;
%     ent_vec(end+1) = ent;
    
    svm_tracker.experts{i}.score(end+1) = loglik - lambda*ent;
    score_temp(i) = sum(svm_tracker.experts{i}.score(max(end,1):end));    
end

% [val idx] = max(score_temp);
svm_tracker.best_expert_idx = numel(score_temp);
if numel(score_temp) >= 2 && config.use_experts
    [val idx] = max(score_temp(1:end-1));
    if score_temp(idx) > score_temp(end) 
        svm_tracker.best_expert_idx = idx;
    end
end

% figure(3)
% for i = 1:n
%     subplot(2,3,i)
%     text(0,1,num2str(svm_tracker.experts{i}.score(end)),'BackgroundColor',[1 1 1]);
%     text(10,1,num2str(score_temp(i)),'BackgroundColor',[1 1 1]);
%     text(0,3,num2str(loglik_vec(i)),'BackgroundColor',[1 1 1]);
%     text(10,3,num2str(ent_vec(i)),'BackgroundColor',[1 1 1]);
% end
% pause

% svm_tracker.output = rect_temp(svm_tracker.best_expert_idx,:);


