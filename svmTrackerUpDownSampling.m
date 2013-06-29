function svmTrackerUpDownSampling(I_vf)
% for scale changes
global sampler
global svm_tracker


updateSample(I_vf,300,1.3);
svmTrackerDo(sampler.patterns_dt);