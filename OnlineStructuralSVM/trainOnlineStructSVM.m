function w = trainOnlineStructSVM(size2train, w_init, find_MVC, param)
%{

This is an online MATLAB implementation of structural SVM with cutting plane algorithm.
This implementation doesn't use sparse structure for the feature and w. 
But it should be straightforward to do that by making use of MATLAB's sparse() function.

Jianxiong Xiao
http://mit.edu/jxiao/

Usage:

Input: 
1. need to have size2train to be the number of training examples
2. need to have w_init as initial values for w. If you have no idea, initialize w by all ones
3. need to have param
   param.C = 1.0;
   param.max_num_iterations = 500;
   param.max_num_constraints =  10000;
4. need to have a function handler @find_MVC to find the most violated constraints.
   This function should have the interfact:
   [X_wo margin] = findMVC_BinaryLinearSVM(w, id)
   1st output X_wo: The constraint corresponding to that labeling = (Groud Truth Feature - Worst Offending feature)
   2nd output m   : the margin you want to enforce for this constraint.   

Output:
    w: the learned SVM weights
%}

w = w_init;

fprintf('------------------trainOnlineStructSVM-------------------\n');
considerIDS = 1:size2train;
Constraints = zeros(numel(w), param.max_num_constraints, 'single');
Margins = zeros(1, param.max_num_constraints, 'single');
IDS = zeros(1, param.max_num_constraints, 'single');
iter=1;
trigger=1;
low_bound= 0;
n=0;
cost = w'*w*.5;
while (iter < param.max_num_iterations && trigger)
    fprintf('Iteration #%d', iter);
    
    %fprintf('time = %s',datestr(now));
    trigger=0;

    for id = considerIDS
        % finds the most violated constraint on example id under the current w
        % 1st output X_wo: The constraint corresponding to that labeling = (Groud Truth Feature - Worst Offending feature)
        % 2nd output m   : the margin you want to enforce for this constraint.        
        [X_wo m]  = find_MVC(w, id);
        
        %if this constraint is the MVC
        isMVC = 1;
        check_labels = find(IDS(1, 1:n) ==id);
        score = m-w'*X_wo;   
        
        for ii=1:numel(check_labels)
            label_ii = check_labels(ii);
            if (m-w'*Constraints(:, label_ii) > score)
                isMVC=0;
                break;
            end
        end    
        
        if isMVC ==1
            cost = cost + param.C*max(0, m - w'*X_wo);
            %add only if this is a hard constraint
            if (m - w'*X_wo) >= -0.001
                n=n+1;
                Constraints(:, n) = X_wo;
                Margins(n) = m;
                IDS(n) = id;

                if n > param.max_num_constraints
                    fprintf('n > param.max_num_constraints');
                    [slacks I_ids] = sort((Margins(:,n)  - w'*Constraints(:, 1:n)), 'descend');
                    J = I_ids(1:param.max_num_constraints);
                    n = length(J);
                    Constraints(:, 1:n) = Constraints(:, J);
                    Margins(:, 1:n) = Margins(:, J);
                    IDS(:, 1:n) = IDS(:, J);

                end

            end
        end
        % fprintf('\n   cost = %f  low_bound = %f', cost, low_bound);

        if 1 - low_bound/cost > .01
            % Call QP
            %if mod(iter, 10) == 1
            %   [cost low_bound]
            %end
            [w,cache]= lsvmopt(Constraints(:,1:n), Margins(1:n), IDS(1:n), param.C, 0.01, []);

            % Prune working set
            I = find(cache.sv > 0);
           
            n = length(I);
            Constraints(:,1:n) = Constraints(:,I);

            Margins(:,1:n) = Margins(:,I);
            IDS(:,1:n) = IDS(:,I);
           
            %reset the running estimate on upper bund
            cost = w'*w*0.5;
            low_bound = cache.lb;
            trigger = 1;
        end        
        
    end
    fprintf('\ntrigger = %d \n---------------------------------------------------------\n',trigger);
    iter = iter +1;
end
