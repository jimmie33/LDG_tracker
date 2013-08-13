function doWork
clc
clear
addpath(genpath('.'));
addpath('../../mexopencv/mexopencv');

input='..\data\david';
D = dir(fullfile(input,'*.png'));
file_list={D.name};

%%for LSH
nbin = 32;            %number of bins
alpha = 0.5;         %parameter of LSH, [0.0,1.0]
k = 0.005;              %parameter of illumination invariant features

%% control parameter
record_vid = false;
image_scale = 1;
max_train_sz = 200;
pixel_step = 6;
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

sam_num = 5;
thr = (1/sam_num:1/sam_num:1-1/sam_num)*255;
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
config.padding = 40;%for object out of border
config.use_raw_feat = false;%do not explode the feature
config.thresh_p = 0.1;
config.thresh_n = 0.5;
config.scale_change = false;
config.lambda = 100;
config.tracking_failure = false;



if record_vid
    vid = avifile('./output/output3.avi');
end

figure(1); set(1,'KeyPressFcn', @handleKey); % open figure for display of result
% figure(2); set(2,'KeyPressFcn', @handleKey); 
figure(3); set(3,'KeyPressFcn', @handleKey); 
finish = 0; 

disagreement_count = 0;
ambiguity_loss = [];
ambiguity_loss_old = [];
amb_count = 0;

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
    if ~config.tracking_failure
        roi = rsz_rt(svm_tracker.output,size(I_scale),config.search_roi);
    else
        roi = rsz_rt(svm_tracker.output,size(I_scale),5*config.search_roi);
    end
    sampler.roi = roi; 
    
    %% crop frame
    I_crop = I_scale(sampler.roi(2):sampler.roi(4),sampler.roi(1):sampler.roi(3),:);
    
    %% compute feature images
tic
    if ~config.tracking_failure
        [BC F] = getFeatureRep(I_crop,config.hist_nbin,config.pixel_step);
    end
%     F = cell2mat(reshape(F,1,1,[]));
toc   
   
   %% tracking part
%    tic
    if frame_id==start_frame
        initSampler(rect,BC,F,pixel_step,use_color);
        train_mask = (sampler.costs<config.thresh_p) | (sampler.costs>=config.thresh_n);
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
        roi_reg = sampler.roi; roi_reg(3:4) = sampler.roi(3:4)-sampler.roi(1:2);
        roi_reg = roi_reg*svm_tracker.scale;
        rectangle('position',roi_reg,'LineWidth',1,'EdgeColor','r');

        
        old_output = svm_tracker.output;
        old_scale = svm_tracker.scale;
        
        if ~config.tracking_failure
            BC = svmTrackerUpDownSampling(BC,F);
        else
            BC = exhausiveSearch(I_crop,[1]);% could evoke a exhausive search
            
            %test to see if tracking is recovered
            if ~config.tracking_failure
%                 config.tracking_failure = false;
                svm_tracker.state = 0;
            else
                svm_tracker.output = old_output;
                svm_tracker.scale = old_scale;
            end
        end
        
        figure(1)
        if svm_tracker.state == 0 && ~config.tracking_failure
            text(0,0,num2str(svm_tracker.confidence));
            rectangle('position',svm_tracker.output*svm_tracker.scale,'LineWidth',2,'EdgeColor','g')
%             rectangle('position',svm_tracker.output_second*svm_tracker.scale,'LineWidth',2,'EdgeColor','b')
        elseif ~config.tracking_failure
            text(0,-5,num2str(svm_tracker.confidence));
            text(0,-20,num2str(svm_tracker.confidence_old));
            rectangle('position',svm_tracker.output*svm_tracker.scale,'LineWidth',2,'EdgeColor','r')
            rectangle('position',svm_tracker.output_old*svm_tracker.scale,'LineWidth',2,'EdgeColor','b')            
        end
       %% update
%         svm_tracker.output_feat_record(end+1,:) = svm_tracker.output_feat_raw;
        svm_tracker.temp_count = svm_tracker.temp_count + 1;
      
        if (svm_tracker.state == 0) && ~config.tracking_failure %check to see if enter mode 1
            temp_diff = mean(abs(svm_tracker.output_feat - sampler.template));
            text(60,0,num2str(temp_diff));
            if svm_tracker.temp_count > 1 && ...
                    ((svm_tracker.ambiguity_loss>5)) % enter mode 1
                % need to snapshot the svm_tracker
                svm_tracker.snapshot = svm_tracker;
                svm_tracker.state = 1;
                diff_hist = temp_diff;   
            else % update sample.template
                sampler.template = 0.95*sampler.template + 0.05*svm_tracker.output_feat;
            end     
            
        elseif ~config.tracking_failure % mode 1: check to see if to return mode 0
            
            curr_diff = mean(abs(svm_tracker.output_feat_old - sampler.template));
            diff_hist(end+1) = curr_diff;
            amb_count = amb_count + 1;
            ambiguity_loss(end+1) = svm_tracker.ambiguity_loss + 0*norm(svm_tracker.w);
            ambiguity_loss_old(end+1) = svm_tracker.ambiguity_loss_old + 0*norm(svm_tracker.snapshot.w);
            if getIOU(svm_tracker.output,svm_tracker.output_old) < 0.8
                disagreement_count = disagreement_count + 1;
            end
%             text(60,-10,[num2str(svm_tracker.confidence_old) ' -- ' num2str(svm_tracker.confidence_old_new)]);
%             
%             
            text(100,-5,num2str(svm_tracker.ambiguity_loss),'color','b');
            text(100,-20,num2str(svm_tracker.ambiguity_loss_old),'color','b');
            
            if ((svm_tracker.confidence < -0.0 && svm_tracker.ambiguity_loss < 2) && ...
                    (svm_tracker.confidence_old < -0.0 && svm_tracker.ambiguity_loss_old < 2)) 
                config.tracking_failure = true;
                svm_tracker = svm_tracker.snapshot;
                svm_tracker.output = old_output;
                svm_tracker.scale = old_scale;
                
                ambiguity_loss = [];
                ambiguity_loss_old = [];
                disagreement_count = 0;
                amb_count = 0;
                keyboard
%             elseif disagreement_count == 0 && amb_count > 20
%                 svm_tracker.state=0;
%                 ambiguity_loss = 0;
%                 ambiguity_loss_old = 0;
%                 disagreement_count = 0;
%                 amb_count = 0;
            elseif sum(ambiguity_loss)  <= sum(ambiguity_loss_old) && ...
                    amb_count > 40                     
                keyboard
                svm_tracker.state=0;
                ambiguity_loss = [];
                ambiguity_loss_old = [];
                disagreement_count = 0;
                amb_count = 0;
%             elseif diff_hist(end) < diff_hist(end-1) && getIOU(svm_tracker.output,svm_tracker.output_old) > 0.8 ...
%                     && svm_tracker.ambiguity_loss == 0 && svm_tracker.confidence > 0
%                 svm_tracker.state=0;  
            elseif  numel(ambiguity_loss)> 9 && ...
                    sum(ambiguity_loss(max(1,end-9):end))  > sum(ambiguity_loss_old(max(1,end-9):end)) 
                    
                pause(2)
                output = svm_tracker.output_old;
                svm_tracker = svm_tracker.snapshot;
                svm_tracker.output = output;
                svm_tracker.state = 0;
                ambiguity_loss = [];
                ambiguity_loss_old = [];
                disagreement_count = 0;
                amb_count = 0;
            
            else
                
%                 sliding_win = struct();
%                 sliding_win.step_size = max(round(sampler.template_size(1:2)/5),1);
%                 feature_map = imresize(BC,1/config.pixel_step,'nearest');
%                 feat_col = im2colstep(feature_map,sampler.template_size,...
%                     [sliding_win.step_size, sampler.template_size(3)]);
%                 sliding_win.patch_size = sampler.template_size;
%                 sliding_win.map_size = size(feature_map);
% %               diff_mask = abs(svm_tracker.snapshot.w) < 0.001;
%                 map_new = getConfidenceMap(feat_col,-svm_tracker.w,-svm_tracker.Bias,sliding_win);
%                 map_old = getConfidenceMap(feat_col,-svm_tracker.snapshot.w,-svm_tracker.snapshot.Bias,sliding_win);
%                 map_diff = -getDifferenceMap(feat_col,sampler.template',sliding_win);
% %                 map_gain = map_new-map_old;
%             
%                 map_new = (map_new-min(map_new(:)))/(max(map_new(:))-min(map_new(:)));
%                 map_old = (map_old-min(map_old(:)))/(max(map_old(:))-min(map_old(:)));
%                 map_diff = (map_diff-min(map_diff(:)))/(max(map_diff(:))-min(map_diff(:)));
% %                 map_gain = (map_gain-min(map_gain(:)))/(max(map_gain(:))-min(map_gain(:)));
%             
% %                 struct_el = strel('square',3);
%                 map_agree = (map_new > 0.95) + (map_old >0.95) + (map_diff>0.95);
% %                 map_new_old = imdilate((map_new > 0.95),struct_el) + imdilate((map_old >0.95),struct_el);
% %               map_new_diff = (map_new > 0.9) + + (map_diff>0.9);
%             
% %                 figure(2)
% %                 subplot(1,4,1)
% %                 imagesc(map_new)
% %                 colorbar
% %                 subplot(1,4,2)
% %                 imagesc(map_old)
% %                 colorbar
% %                 subplot(1,4,3)
% %                 imagesc(map_diff)
% %                 colorbar
% %                 subplot(1,4,4)
% %                 imagesc(map_agree)
% %                 colorbar
% %                 figure(1)
% 
%             
%                 if (getIOU(svm_tracker.output,svm_tracker.output_old) < 0.5 && ... % check tracking failure
%                         max(map_agree(:)) == 1 && svm_tracker.confidence_old < 0) ||...
%                         svm_tracker.confidence < -1 || ...
%                         svm_tracker.confidence_old < -1
%                     config.tracking_failure = true;
%                     svm_tracker = svm_tracker.snapshot;
%                     svm_tracker.output = old_output;
%                     svm_tracker.scale = old_scale;
%                     keyboard
%                 
% %                 elseif getIOU(svm_tracker.output,svm_tracker.output_old) < 0.8 && ...
% %                             diff_hist(end) < diff_hist(end-1) && max(map_agree(:)) > 1 && svm_tracker.confidence_old>0
% % %                       svm_tracker.confidence_old>0 && svm_tracker.confidence_old_new >0% restore
% %                         output = svm_tracker.output_old;
% %                         svm_tracker = svm_tracker.snapshot;
% %                         svm_tracker.output = output;
% %                         svm_tracker.state = 0;                
%       
%                 end  
            
            end
        end 
        
        
tic
        if (svm_tracker.confidence > 0 || svm_tracker.ambiguity_loss > 2) && ~config.tracking_failure
            resample(BC,300,1.5);        
            train_mask = (sampler.costs<config.thresh_p) | (sampler.costs>=config.thresh_n);
            label = sampler.costs(train_mask) < config.thresh_p;
            
            skip_train = false;
            if svm_tracker.confidence > 1.0 
                score_ = -(sampler.patterns_dt(train_mask,:)*svm_tracker.w'+svm_tracker.Bias);
                if prod(double(score_(label) > 1)) == 1 && prod(double(score_(~label)<1)) == 1
                    skip_train = true;
                    disp('skipped training!!!!!!!!!!!!!!!!')
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
%        subplot(1,2,1)
%        imshow(1-F{1});
%        subplot(1,2,2)
%        imshow(F{2});
% %        subplot(1,3,3)
% %        imshow(F{3});
%    else
%        subplot(1,4,1)
%        imshow(1-F{1});
%        subplot(1,4,2)
%        imshow(F{2});
%        subplot(1,4,3)
%        imshow(F{3});
%        subplot(1,4,4)
%        imshow(F{4});
% %        subplot(1,5,5)
% %        imshow(F{5});
%    end
       
   
   figure(3)

   subplot(2,2,1)
   output = svm_tracker.output*svm_tracker.scale;
   imshow(I_orig(round(output(2)+1:output(2)+output(4)-1),...
       round(output(1)+1:output(1)+output(3)-1),:));
   subplot(2,2,2) % visualize svm weight vector
   svm_w = abs(reshape(svm_tracker.w,sampler.template_size(1),sampler.template_size(2),[]));
   imagesc(sum(svm_w,3));
   subplot(2,2,3:4)
   plot(sum(abs(reshape(svm_w,sampler.template_size(1)*sampler.template_size(2),[])),1),'go')
   
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