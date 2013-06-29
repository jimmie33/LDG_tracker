function tracker = createSvmTracker()

tracker.sv_size = 500;% maxial 100 cvs
tracker.C = 10000;
tracker.B = 80;% for tvm
tracker.B_p = 10;% for positive sv
tracker.lambda = 1;% for whitening
tracker.m1 = 1;% for tvm
tracker.m2 = 1.5;% for tvm
tracker.w = [];
tracker.w_smooth_rate = 0.8;
tracker.confidence = 1;
tracker.scale = 1;
tracker.scale_step = 1.05;
tracker.scale_upbound = 2;
tracker.scale_lowbound = 0.5;
tracker.solver = 5; %0: matlab built-in
                    %1: liblinear
                    %2: libsvm
                    %3: srsvm
                    %4: psvm
                    %5: tvm