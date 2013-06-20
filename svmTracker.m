function tracker = svmTracker()

tracker.sv_size = 500;% maxial 100 cvs
tracker.sigma = 1;%rbf
tracker.C = 1;
tracker.solver = 0; %0: matlab built-in
                    %1: liblinear
                    %2: libsvm
                    %3: struct-svm