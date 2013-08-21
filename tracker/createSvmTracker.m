function tracker = createSvmTracker()

tracker.sv_size = 500;% maxial 100 cvs
tracker.C = 100;
tracker.B = 80;% for tvm
tracker.B_p = 10;% for positive sv
tracker.lambda = 1;% for whitening
tracker.m1 = 1;% for tvm
tracker.m2 = 2;% for tvm
tracker.w = [];
tracker.w_smooth_rate = 0.0;
tracker.confidence = 1;
tracker.scale = 1;
tracker.state = 0;
tracker.temp_count = 0;
tracker.output_feat_record = [];
tracker.feat_cache = [];
tracker.experts = {};
tracker.max_expert_sz = 6;
tracker.expert_update_interval = 25;
tracker.confidence_exp = 1;
tracker.confidence = 1;
tracker.best_expert_idx = 1;
tracker.solver = 5; %0: matlab built-in
                    %1: liblinear
                    %2: libsvm
                    %3: srsvm
                    %4: psvm
                    %5: tvm