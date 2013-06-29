function initSampler(init_rect,I_vf,step,use_color)
global sampler;

sampler.step = step;
% sampler.state=repmatls(init_rect,[size(sampler.state,1),1]);
% sampler.state(2:end,1:2) = sampler.state(2:end,1:2)+(rand(size(sampler.state,1)-1,2)-0.5).*(sampler.radius*2*max(sampler.state(1,3:4)));

init_rect_roi = init_rect;
init_rect_roi(1:2) = init_rect(1:2) - sampler.roi(1:2)+1;
sampler.template = I_vf (round(init_rect_roi(2):sampler.step:init_rect_roi(2)+init_rect_roi(4)),...
    round(init_rect_roi(1):sampler.step:init_rect_roi(1)+init_rect_roi(3)),:);
sampler.template_width = init_rect(3);
sampler.template_height = init_rect(4);
if use_color
    sampler.feature_num = 4;
else
    sampler.feature_num = 2;
end

%% for collecting initial training data
resample(I_vf,300,1.5);




