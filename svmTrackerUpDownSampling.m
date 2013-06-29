function I_vf = svmTrackerUpDownSampling(I_vf)
% for scale changes
global sampler
global svm_tracker


updateSample(I_vf,300,1.3);
svmTrackerDo(sampler.patterns_dt);
confidence_orig = svm_tracker.confidence;
scale_orig = svm_tracker.scale;
output_orig = svm_tracker.output;
roi_orig = sampler.roi;

%scale up
if scale_orig < svm_tracker.scale_upbound
    svm_tracker.scale = scale_orig*svm_tracker.scale_step;
    sampler.roi = roi_orig + ...
        repmat(0.5*(1/svm_tracker.scale_step-1)*(roi_orig(1:2)+roi_orig(3:4)),[1,2]);
    I_vf_up = imresize(I_vf,1/svm_tracker.scale_step);
    updateSample(I_vf_up,300,1.3);
    svmTrackerDo(sampler.patterns_dt);
    confidence_up = svm_tracker.confidence;
    scale_up = svm_tracker.scale;
    output_up = svm_tracker.output;
    roi_up = sampler.roi;
else
    confidence_up = -inf;
end

%scale down
if scale_orig > svm_tracker.scale_lowbound
    svm_tracker.scale = scale_orig/svm_tracker.scale_step;
    sampler.roi = roi_orig + ...
        repmat(0.5*(svm_tracker.scale_step-1)*(roi_orig(1:2)+roi_orig(3:4)),[1,2]);
    I_vf_down = imresize(I_vf,svm_tracker.scale_step);
    updateSample(I_vf_down,300,1.3);
    svmTrackerDo(sampler.patterns_dt);
    confidence_down = svm_tracker.confidence;
    scale_down = svm_tracker.scale;
    output_down = svm_tracker.output;
    roi_down = sampler.roi;
else
    confidence_down = -inf;
end


% if confidence_up >= confidence_orig
%     keyboard
% end

if confidence_up > confidence_down && ...
        (confidence_up > confidence_orig + 0.1)
    svm_tracker.confidence = confidence_up;
    svm_tracker.output = output_up;
    svm_tracker.scale = scale_up;
    sampler.roi = roi_up;
    I_vf = I_vf_up;
elseif ~( confidence_down > confidence_up && ...
        confidence_down >confidence_orig + 0.1)
    svm_tracker.confidence = confidence_orig;
    svm_tracker.output = output_orig;
    svm_tracker.scale = scale_orig;
    sampler.roi = roi_orig;
else
    I_vf = I_vf_down;
end

%scale down