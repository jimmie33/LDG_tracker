function evaluateExperts(I_vf,lambda,sigma)
global sampler;
global svm_tracker;
global experts;
global config;

feature_map = imresize(I_vf,1/config.pixel_step,'nearest');
step_size = [1,1];%max(round(sampler.template_size(1:2)/10),1);

if size(I_vf,3)>1
    patterns_dt = im2colstep(feature_map,sampler.template_size,[step_size, size(I_vf,3)]);
else
    patterns_dt = im2colstep(feature_map,sampler.template_size,step_size);
end

rects = repmat(svm_tracker.output,[size(patterns_dt,2),1]);

x = 1:step_size(2):size(feature_map,2)-sampler.template_size(2)+1;
y = 1:step_size(1):size(feature_map,1)-sampler.template_size(1)+1;
[X Y] = meshgrid(x,y);
rects(1:end,1) = (X(:)-1)*config.pixel_step + sampler.roi(1);
rects(1:end,2) = (Y(:)-1)*config.pixel_step + sampler.roi(2);

% mask_temp(1:5:end,1:5:end) = 1;
% nrow = numel(1:5:numel(y));
% ncol = numel(1:5:numel(x));
label_prior = ones([numel(y),numel(x)]);%fspecial('gaussian',[numel(y),numel(x)],sigma);
% interval = 4;


%% compute log likelihood and entropy
n = numel(experts);
score_temp = zeros(n,1);
rect_temp = zeros(n,4);
% figure(4)
% 
if config.debug
    loglik_vec=[];
    ent_vec=[];
    figure(3)
end

kernel_size = sampler.template_size(1:2);%half template size;

% svm_tracker.experts{2}.w = 0.99*svm_tracker.experts{2}.w + 0.01*svm_tracker.experts{3}.w;
% svm_tracker.experts{2}.Bias = 0.99*svm_tracker.experts{2}.Bias + 0.01*svm_tracker.experts{3}.Bias;
mask_temp = zeros(numel(y),numel(x));
idx_temp = [];
svm_scores = [];
svm_score = {};
for i = 1:n
    svm_score{i} = -(experts{i}.w*patterns_dt+experts{i}.Bias);
    [val idx] = max(svm_score{i});
    best_rect = rects(idx,:);
    rect_temp(i,:) = best_rect;
    svm_scores(i) = val;
    idx_temp(i) = idx;
    [r c] = ind2sub(size(mask_temp),idx);
    mask_temp(r,c) = 1;
end
perturb = (1:numel(mask_temp))/(numel(mask_temp)*100000);
mask_temp_blur = imfilter(mask_temp+reshape(perturb,size(mask_temp,1),[]),fspecial('gaussian',round(kernel_size)));
mask_temp = mask_temp_blur == imdilate(mask_temp_blur,strel('rectangle',round(kernel_size*0.5)));
mask_temp = mask_temp & mask_temp_blur > 1/100000;
mask_temp = mask_temp > 0;
for i = 1:n
    mask = mask_temp(:);
    [loglik ent] = getLogLikelihoodEntropy(svm_score{i}(mask(:)),label_prior(mask(:)));
    if config.debug
        loglik_vec(end+1) = loglik;
        ent_vec(end+1) = ent;
        subplot(2,3,i)    
        imagesc(reshape(svm_score{i},numel(y),[]));
        colorbar
    end
    
    experts{i}.score(end+1) = loglik - lambda*ent;
    score_temp(i) = sum(experts{i}.score(max(end+1-config.entropy_score_winsize,1):end));    
end

% [val idx] = max(score_temp);
% svm_tracker.failure = false;
svm_tracker.best_expert_idx = numel(score_temp);
if numel(score_temp) >= 2 && config.use_experts
    [val idx] = max(score_temp(1:end-1));
%     if score_temp(idx) > score_temp(end) && svm_scores(idx) < 0
%         svm_tracker.failure = true;
%     end
    if score_temp(idx) > score_temp(end) && svm_scores(idx) > config.svm_thresh
        % recover previous version
        output = svm_tracker.output;
        experts{end}.snapshot = svm_tracker;
        svm_tracker = experts{idx}.snapshot;
        svm_tracker.output = output;
        svm_tracker.best_expert_idx = idx;
%         experts([idx end]) = experts([end idx]);
    end
end
% svm_tracker.w = experts{svm_tracker.best_expert_idx}.w;
% svm_tracker.Bias = experts{svm_tracker.best_expert_idx}.Bias;

 
if config.debug

    for i = 1:n
        subplot(2,3,i)
        if i == svm_tracker.best_expert_idx
            color = [1 0 0];
        else
            color = [1 1 1];
        end
        text(0,1,num2str(experts{i}.score(end)),'BackgroundColor',color);
        text(15,1,num2str(score_temp(i)),'BackgroundColor',color);
        text(0,3,num2str(loglik_vec(i)),'BackgroundColor',color);
        text(15,3,num2str(ent_vec(i)),'BackgroundColor',color);
    end
    subplot(2,3,6)
    imagesc(mask_temp)
    figure(1)
end
% pause

% if svm_scores(svm_tracker.best_expert_idx)>0
%     svm_tracker.output = rect_temp(svm_tracker.best_expert_idx,:);
% end


