function doWork
clc
clear
addpath(genpath('.'));
addpath('../../mexopencv/mexopencv');

input='..\data\lemming';
D = dir(fullfile(input,'*.jpg'));
file_list={D.name};

%%for LSH
nbin = 32;            %number of bins
alpha = 0.5;         %parameter of LSH, [0.0,1.0]
k = 0.005;              %parameter of illumination invariant features

%% control parameter
record_vid = false;
image_scale = 0.7;
max_train_sz = 200;
pixel_step = 5;
use_color = true;
search_roi = 2.4; % the ratio of the search radius to the longest edge of bbox
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

sam_num = 5;
thr = (1/sam_num:1/sam_num:1-1/sam_num)*255;
fd = length(thr);
patterns={};
costs={};
medscore=zeros(1,visualize_medscore_size);

% config
global config
global finish %flag for determination

if record_vid
    vid = avifile('./output/output3.avi');
end

figure(1); set(1,'KeyPressFcn', @handleKey); % open figure for display of result
% figure(2); set(2,'KeyPressFcn', @handleKey); 
figure(3); set(3,'KeyPressFcn', @handleKey); 
finish = 0; 


for frame_id=start_frame:numel(file_list)
    
%     disp('**********************')
    if finish == 1
        break;
    end
    
    %% read a frame
    I_orig=imread(fullfile(input,file_list{frame_id}));
    %% intialization
    if frame_id==start_frame
        figure(1)
        imshow(I_orig);
        % crop to get the initial window
        [InitPatch rect]=imcrop(I_orig); rect = round(rect);%
        disp(rect);
        config = makeConfig(I_orig,rect);
        svm_tracker.output = rect*config.image_scale;
        svm_tracker.output(1:2) = svm_tracker.output(1:2) + config.padding;
        svm_tracker.output_exp = svm_tracker.output;
    end
    [I_orig]= getFrame2Compute(I_orig);
    %% compute ROI and scale image
    I_scale = cv.resize(I_orig,1/svm_tracker.scale);
    if frame_id == start_frame
        roi = rsz_rt(svm_tracker.output,size(I_scale),6);
    elseif svm_tracker.confidence > 0
        roi = rsz_rt(svm_tracker.output,size(I_scale),config.search_roi);
    else % use the same roi
%         roi = rsz_rt(svm_tracker.output,size(I_scale),5*config.search_roi);
    end
    sampler.roi = roi; 
    
    %% crop frame
    I_crop = I_scale(sampler.roi(2):sampler.roi(4),sampler.roi(1):sampler.roi(3),:);
    
    %% compute feature images
% tic
    [BC F] = getFeatureRep(I_crop,config.hist_nbin,config.pixel_step);

%     F = cell2mat(reshape(F,1,1,[]));
% toc   
   
   %% tracking part
%    tic
    if frame_id==start_frame
        initSampler(svm_tracker.output,BC,F,config.pixel_step,config.use_color);
        train_mask = (sampler.costs<config.thresh_p) | (sampler.costs>=config.thresh_n);
        label = sampler.costs(train_mask,1)<config.thresh_p;
        costs = sampler.costs(train_mask);
        fuzzy_weight = ones(size(label));
%         fuzzy_weight(~label) = 2*costs(~label)-1;
% tic
        initSvmTracker(sampler.patterns_dt(train_mask,:), label, fuzzy_weight);
% toc
        figure(1);
        imshow(I_orig);
        rectangle('position',svm_tracker.output,'LineWidth',2,'EdgeColor','b')
    else
        % testing
       
        figure(1)
        imshow(I_orig);       
        roi_reg = sampler.roi; roi_reg(3:4) = sampler.roi(3:4)-sampler.roi(1:2);
        roi_reg = roi_reg*svm_tracker.scale;
        rectangle('position',roi_reg,'LineWidth',1,'EdgeColor','r');

        
        old_output = svm_tracker.output;
        old_scale = svm_tracker.scale;
        
        
        
        
        if true
            if mod(frame_id-start_frame+1,svm_tracker.expert_update_interval) == 0
                updateTrackerExperts;
            end
            evaluateExperts(BC,100);
            
            BC = svmTrackerUpDownSampling(BC,F);
            
            % visualize sampling positions
%             pos_train = sampler.state_dt;
%             for k = 1:size(pos_train,1)
%                 px = pos_train(k,1)+0.5*pos_train(k,3);
%                 py = pos_train(k,2)+0.5*pos_train(k,4);
%                 rectangle('position',[px,py,1,1],'LineWidth',1,'EdgeColor','r')
%             end 

        end
        
        figure(1)
        if svm_tracker.state == 0 
            text(0,-5,num2str(svm_tracker.confidence_exp));
            text(0,-20,num2str(svm_tracker.confidence));
            rectangle('position',svm_tracker.output_exp*svm_tracker.scale,'LineWidth',2,'EdgeColor','g')
            rectangle('position',svm_tracker.output*svm_tracker.scale,'LineWidth',2,'EdgeColor','r')
%             if svm_tracker.confidence ~= svm_tracker.confidence_cur
% %                 keyboard
%             end
%             rectangle('position',svm_tracker.output_second*svm_tracker.scale,'LineWidth',2,'EdgeColor','b')
%         elseif ~config.tracking_failure
%             text(0,-5,num2str(svm_tracker.confidence));
% %             text(0,-20,num2str(svm_tracker.confidence_old));
%             rectangle('position',svm_tracker.output*svm_tracker.scale,'LineWidth',2,'EdgeColor','r')
%             rectangle('position',svm_tracker.output_old*svm_tracker.scale,'LineWidth',2,'EdgeColor','b')            
        end
        
 
       %% update
%         svm_tracker.output_feat_record(end+1,:) = svm_tracker.output_feat_raw;
        svm_tracker.temp_count = svm_tracker.temp_count + 1;
      
        

        
        
% tic
        if (frame_id - start_frame < 25 && svm_tracker.confidence >-1) ||...
                (svm_tracker.confidence_exp ~= svm_tracker.confidence && svm_tracker.confidence_exp > 0) ||...
                (svm_tracker.confidence_exp == svm_tracker.confidence && svm_tracker.confidence_exp > -0.5)
            resample(BC);        
            train_mask = (sampler.costs<config.thresh_p) | (sampler.costs>=config.thresh_n);
            label = sampler.costs(train_mask) < config.thresh_p;
            
            skip_train = false;
            if svm_tracker.confidence > 1.0 
                score_ = -(sampler.patterns_dt(train_mask,:)*svm_tracker.w'+svm_tracker.Bias);
                if prod(double(score_(label) > 1)) == 1 && prod(double(score_(~label)<1)) == 1
                    skip_train = true;
%                     disp('skipped training!!!!!!!!!!!!!!!!')
                end
            end
            
            if ~skip_train
                costs = sampler.costs(train_mask);
                fuzzy_weight = ones(size(label));
                fuzzy_weight(~label) = 2*costs(~label)-1;
%             fuzzy_weight(label) = 1-2*costs(label);
                updateSvmTracker (sampler.patterns_dt(train_mask,:),label,fuzzy_weight);  
                text(250,0,num2str(svm_tracker.margin),'color','r');
            end
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
% toc
    end
    
    drawnow
%    toc
   %% visulize results

%    fig=figure(1);
%    if record_vid
%        Fr = getframe(fig);
%        vid = addframe(vid,Fr);
%    end
%    
%    
%    figure(3)
% 
%    subplot(2,2,1)
%    output = svm_tracker.output*svm_tracker.scale;
%    imshow(I_orig(round(output(2)+1:output(2)+output(4)-1),...
%        round(output(1)+1:output(1)+output(3)-1),:));
%    subplot(2,2,2) % visualize svm weight vector
%    svm_w = abs(reshape(svm_tracker.w,sampler.template_size(1),sampler.template_size(2),[]));
%    imagesc(sum(svm_w,3));
%    subplot(2,2,3:4)
%    plot(sum(abs(reshape(svm_w,sampler.template_size(1)*sampler.template_size(2),[])),1),'go')
%    
%    figure(2)
%    temp = [];
%    for k = 1:numel(svm_tracker.experts)
%        temp = [temp;svm_tracker.experts{k}.score(max(1,end-24):end)];
%    end
%    plot(temp'),hold on
%    if ~isempty(temp)
%        plot(temp(end,:),'g')
%    end
%    hold off
end
if record_vid
    vid = close(vid);
end