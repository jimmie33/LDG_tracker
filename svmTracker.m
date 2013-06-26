function tracker = svmTracker()

tracker.sv_size = 500;% maxial 100 cvs
tracker.C = 1000;
tracker.B = 50;% for tvm
tracker.B_p = 10;% for positive sv
tracker.m1 = 1;% for tvm
tracker.m2 = 2;% for tvm
tracker.solver = 5; %0: matlab built-in
                    %1: liblinear
                    %2: libsvm
                    %3: srsvm
                    %4: psvm
                    %5: tvm