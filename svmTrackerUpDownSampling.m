function I_vf = svmTrackerUpDownSampling(I_vf,I)
% for scale changes
global sampler
global svm_tracker
global config


updateSample(I_vf,I,500,1);
% svmTrackerDo(sampler.patterns_dt);
% updateSample(I_vf,50,0.1);
% svmTrackerDo(sampler.patterns_dt);

if config.scale_change %&& svm_tracker.state == 0
    
    confidence_orig = svm_tracker.confidence;
    scale_orig = svm_tracker.scale;
    output_orig = svm_tracker.output;
    if svm_tracker.state == 1
        output_old_orig = svm_tracker.output_old;
    end
    roi_orig = sampler.roi;

%scale up
    if scale_orig < svm_tracker.scale_upbound
        svm_tracker.scale = scale_orig*svm_tracker.scale_step;
        sampler.roi = roi_orig/svm_tracker.scale_step;
%         roi_orig + ...
%             repmat(0.5*(1/svm_tracker.scale_step-1)*(roi_orig(1:2)+roi_orig(3:4)),[1,2]);
        I_vf_up = imresize(I_vf,1/svm_tracker.scale_step);
        updateSample(I_vf_up,I,100,0.3);
%         svmTrackerDo(sampler.patterns_dt);
        confidence_up = svm_tracker.confidence;
        scale_up = svm_tracker.scale;
        output_up = svm_tracker.output;
        if svm_tracker.state == 1
            output_old_up = svm_tracker.output_old;
        end
        roi_up = sampler.roi;
    else
        confidence_up = -inf;
    end

    %scale down
    if scale_orig > svm_tracker.scale_lowbound
        svm_tracker.scale = scale_orig/svm_tracker.scale_step;
        sampler.roi = roi_orig*svm_tracker.scale_step;
%         roi_orig + ...
%             repmat(0.5*(svm_tracker.scale_step-1)*(roi_orig(1:2)+roi_orig(3:4)),[1,2]);
        I_vf_down = imresize(I_vf,svm_tracker.scale_step);
        updateSample(I_vf_down,I,100,0.3);
%         svmTrackerDo(sampler.patterns_dt);
        confidence_down = svm_tracker.confidence;
        scale_down = svm_tracker.scale;
        output_down = svm_tracker.output;
        if svm_tracker.state == 1
            output_old_down = svm_tracker.output_old;
        end
        roi_down = sampler.roi;
    else
        confidence_down = -inf;
    end


% if confidence_up >= confidence_orig
%     keyboard
% end

    if confidence_up > confidence_down && ...
            (confidence_up > confidence_orig + 0.3 && confidence_up > 0.5)
        svm_tracker.confidence = confidence_up;
        svm_tracker.output = output_up;
        if svm_tracker.state == 1
            svm_tracker.output_old = output_old_up;
        end
        svm_tracker.scale = scale_up;
        sampler.roi = roi_up;
        I_vf = I_vf_up;
    elseif ~( confidence_down > confidence_up && ...
            confidence_down >confidence_orig + 0.3 && confidence_down > 0.5)
        svm_tracker.confidence = confidence_orig;
        svm_tracker.output = output_orig;
        if svm_tracker.state == 1
            svm_tracker.output_old = output_old_orig;
        end
        svm_tracker.scale = scale_orig;
        sampler.roi = roi_orig;
    else
        I_vf = I_vf_down;
    end
    
end

%scale down