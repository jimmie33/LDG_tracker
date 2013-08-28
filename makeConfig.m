function config = makeConfig(frame,selected_rect)

%% automatic setting up

% check if the frame is in RGB format
config.use_color = false;
if (size(frame,3) == 3 && ~isequal(frame(:,:,1),frame(:,:,2),frame(:,:,3)))
    config.use_color = true;    
end

% decide feature channel number
if config.use_color
    thr_n = 5; 
else
    thr_n = 9;
end
config.thr = (1/thr_n:1/thr_n:1-1/thr_n)*255;
config.fd = numel(config.thr);

% decide image scale and pixel step for sampling feature
% rescale raw input frames propoerly would save much computation 
frame_min_width = 320;
trackwin_max_dimension = 64;
template_max_numel = 144;
% min_pixel_step = 2;
frame_sz = size(frame);

if max(selected_rect(3:4)) <= trackwin_max_dimension ||...
        frame_sz(2) <= frame_min_width
    config.image_scale = 1;
else
    min_scale = frame_min_width/frame_sz(2);
    config.image_scale = max(trackwin_max_dimension/max(selected_rect(3:4)),min_scale);    
end
wh_rescale = selected_rect(3:4)*config.image_scale;
win_area = prod(wh_rescale);
config.ratio = (sqrt(template_max_numel/win_area));
template_sz = round(wh_rescale*config.ratio); 
config.template_sz = template_sz([2 1]);

% pixel_step = ceil(sqrt(win_area/template_max_numel));
% config.pixel_step = 1/config;
%% default setting up

config.debug = false;
config.verbose = false;
config.display = true; % show tracking result at runtime
config.scale_change = false;
config.use_experts = true;
config.use_raw_feat = false; % raw intensity feature value
config.use_iif = true; % use illumination invariant feature

config.svm_thresh = -0.7; % for detecting the tracking failure
config.max_expert_sz = 4;
config.expert_update_interval = 50;
config.update_count_thresh = 10;
config.entropy_score_winsize = 5;
config.expert_lambda = 1;

config.search_roi = 2; % ratio of the search roi to tracking window 1.3
config.padding = 40; % for object out of border

config.hist_nbin = 32; % histogram bins for iif computation

config.thresh_p = 0.1; % IOU threshold for positive training samples
config.thresh_n = 0.5; % IOU threshold for negative ones


config.ellipse_mask = false; %mask the template with an ellipse;



config.scale_step = 1.2;
config.scale_upbound = 1.5;
config.scale_lowbound = 0.7;

