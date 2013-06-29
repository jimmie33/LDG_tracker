clc
clear
addpath(genpath('.'));
addpath('../../mexopencv/mexopencv');

input='..\data\Freeman3\img\';
D = dir(fullfile(input,'*.jpg'));
file_list={D.name};

%%for LSH
nbin = 32;            %number of bins
alpha = 0.5;         %parameter of LSH, [0.0,1.0]
k = 0.005;              %parameter of illumination invariant features

%% control parameter
record_vid = false;
image_scale = 1;
max_train_sz = 200;
pixel_step = 3;
use_color = false;
search_roi = 3; % the ratio of the search radius to the longest edge of bbox
init_step = 20;
start_frame = 1;


visualize_medscore_size = 200;


%% counters
sample_count = 0;
initialized = true;


%%%%%%%%%%%%%%%%%%%%%%%
global sampler;
global svm_tracker;

sampler=createSampler();
svm_tracker = createSvmTracker();

sam_num = 9;
thr = 1/sam_num:1/sam_num:1-1/sam_num;
fd = length(thr);
patterns={};
costs={};
medscore=zeros(1,visualize_medscore_size);

% config
global config
global finish %flag for determination
config.verbose = false;
config.image_scale = image_scale;
config.use_color = use_color;
config.search_roi = search_roi;
config.hist_decay = 0.1;
config.hist_nbin = 32;
config.IIF_k = 0.005;
config.fd = fd;
config.thr =thr;
config.pixel_step = pixel_step;
config.padding = 20;%for object out of border
config.use_raw_feat = false;%do not explode the feature
config.thresh_p = 0.1;
config.thresh_n = 0.5;



if record_vid
    vid = avifile('./output/output.avi');
end

figure(1); set(1,'KeyPressFcn', @handleKey); % open figure for display of result
figure(3); set(3,'KeyPressFcn', @handleKey); 
finish = 0; 

for frame_id=start_frame:numel(file_list)
    
    disp('**********************')
    if finish == 1
        break;
    end
    
    %% read a frame
    I_orig=imread(fullfile(input,file_list{frame_id}));     
    [I_orig]= getFrame2Compute(I_orig);
    
    %% intialize a bbox
    if frame_id==start_frame
        figure(1)
        imshow(I_orig);
        % crop to get the initial window
        [InitPatch rect]=imcrop(I_orig); rect = round(rect);%
        svm_tracker.output = rect;
    end
    
    %% compute ROI and scale image
    I_scale = cv.resize(I_orig,1/svm_tracker.scale);
    roi = rsz_rt(svm_tracker.output,size(I_scale),config.search_roi);
    sampler.roi = roi; 
    
    %% crop frame
    I_crop = I_scale(sampler.roi(2):sampler.roi(4),sampler.roi(1):sampler.roi(3),:);
    
    %% compute feature images
tic
    alpha = exp(-sqrt(2)/(config.hist_decay*min(svm_tracker.output(3:4))));
    [BC F] = getFeatureRep(I_crop,alpha,config.hist_nbin,config.IIF_k,config.pixel_step);
toc   
   
   %% tracking part
%    tic
    if frame_id==start_frame
        initSampler(rect,BC,pixel_step,use_color);
        train_mask = (sampler.costs<config.thresh_p) | (sampler.costs>config.thresh_n);
        label = sampler.costs(train_mask,1)<config.thresh_p;
tic
        initSvmTracker(sampler.patterns_dt(train_mask,:), label);
toc
        figure(1);
        imshow(I_orig);
        rectangle('position',svm_tracker.output,'LineWidth',2,'EdgeColor','b')
    else
        % testing
       
        figure(1)
        imshow(I_orig);       
        roi_reg = sampler.roi; roi_reg(3:4) = roi(3:4)-roi(1:2);
        roi_reg = roi_reg*svm_tracker.scale;
        rectangle('position',roi_reg,'LineWidth',1,'EdgeColor','r');
       
        %correct sampler and label
tic
        updateSample(BC);
toc
        svmTrackerDo(sampler.patterns_dt);
% %         rectangle('position',svm_tracker.output,'LineWidth',2,'EdgeColor','b')
%         svmTrackerUpDownSampling(BC);
        
        if svm_tracker.confidence > -1
            text(0,0,num2str(svm_tracker.confidence));
            rectangle('position',svm_tracker.output,'LineWidth',2,'EdgeColor','g')
        else
            text(0,0,num2str(svm_tracker.confidence));
            rectangle('position',svm_tracker.output,'LineWidth',2,'EdgeColor','r')
        end
tic
        if svm_tracker.confidence > -1
            resample(BC);        
            train_mask = (sampler.costs<config.thresh_p) | (sampler.costs>config.thresh_n);
            label = sampler.costs(train_mask,1)<config.thresh_p;       
            updateSvmTracker (sampler.patterns_dt(train_mask,:),label);            
            %% visualize traing sample
%             pos_train = sampler.state_dt(train_mask,:);
%             for k = 1:size(label,1)
%                 px = pos_train(k,1)+0.5*pos_train(k,3);
%                 py = pos_train(k,2)+0.5*pos_train(k,4);
%                 if label(k)>0
%                     rectangle('position',[px,py,1,1],'LineWidth',1,'EdgeColor','g')
%                 else
%                     rectangle('position',[px,py,1,1],'LineWidth',1,'EdgeColor','r')
%                 end
%             end 
        end
toc
    end
%    toc
   %% visulize results

   fig=figure(1);
   if record_vid
       Fr = getframe(fig);
       vid = addframe(vid,Fr);
   end
   
%    figure(2)
%    if ~use_color
%        subplot(1,3,1)
%        imshow(1-F{1});
%        subplot(1,3,2)
%        imshow(F{2});
%        subplot(1,3,3)
%        imshow(F{3});
%    else
%        subplot(1,5,1)
%        imshow(1-F{1});
%        subplot(1,5,2)
%        imshow(F{2});
%        subplot(1,5,3)
%        imshow(F{3});
%        subplot(1,5,4)
%        imshow(F{4});
%        subplot(1,5,5)
%        imshow(F{5});
%    end
       
   
   figure(3)

   subplot(2,2,1)
   imshow(I_orig(round(svm_tracker.output(2):svm_tracker.output(2)+svm_tracker.output(4)-1),...
       round(svm_tracker.output(1):svm_tracker.output(1)+svm_tracker.output(3)-1),:));
   subplot(2,2,2) % visualize svm weight vector
   svm_w = abs(reshape(svm_tracker.w,size(sampler.template,1),size(sampler.template,2),[]));
   imagesc(sum(svm_w,3));
   subplot(2,2,3:4)
   plot(sum(abs(reshape(svm_w,size(sampler.template,1)*size(sampler.template,2),[])),1),'go')
   
%    if svm_tracker.solver == 5
%        fig = figure(5);
%        clf(fig);
%        sv_num = size(svm_tracker.pos_sv,1);
%        pos_score_sv = -(svm_tracker.pos_sv*svm_tracker.clsf.w'+svm_tracker.clsf.Bias);
%        for i = 1:sv_num
%            sv = reshape(svm_tracker.pos_sv(i,:),size(sampler.template,1),...
%                size(sampler.template,2),[]);
%            sv = sv(:,:,3);
%            subplot(1,sv_num,i)
%            imshow(sv);
%            text(0,0,num2str(pos_score_sv(i)));
%        end
% %        pause
%    end

   
end
if record_vid
    vid = close(vid);
end