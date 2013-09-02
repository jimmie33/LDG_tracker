function initSampler(init_rect,I_vf,I,use_color)
global config
global sampler;

% sampler.step = step;
% sampler.state=repmatls(init_rect,[size(sampler.state,1),1]);
% sampler.state(2:end,1:2) = sampler.state(2:end,1:2)+(rand(size(sampler.state,1)-1,2)-0.5).*(sampler.radius*2*max(sampler.state(1,3:4)));

init_rect_roi = init_rect;
init_rect_roi(1:2) = init_rect(1:2) - sampler.roi(1:2)+1;
template = I_vf (round(init_rect_roi(2):init_rect_roi(2)+init_rect_roi(4)-1),...
    round(init_rect_roi(1):init_rect_roi(1)+init_rect_roi(3)-1),:);
sampler.template = imresize(template,config.template_sz);
sampler.template_size = size(sampler.template);
sampler.template = sampler.template(:)';
% sampler.template_raw = I (round(init_rect_roi(2):sampler.step:init_rect_roi(2)+init_rect_roi(4)-1),...
%     round(init_rect_roi(1):sampler.step:init_rect_roi(1)+init_rect_roi(3)-1),:);
% sampler.template_raw = sampler.template_raw(:)';
sampler.template_width = init_rect(3);
sampler.template_height = init_rect(4);
if use_color
    sampler.feature_num = 4;
else
    sampler.feature_num = 2;
end
% sampler.hash_function.a = randi([1,numel(sampler.template_raw)],8,24);
% sampler.hash_function.b = rand(8,24)*255;
% sampler.hash_table = sparse(8,2^24);
sampler.mask = ~getEllipseMask(sampler.template_size);

%% for collecting initial training data
resample(I_vf);




