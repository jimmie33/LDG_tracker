function [X_wo margin] = findMVC_BinaryLinearSVM(w, id)
    % finds the most violated constraint on example id under the current w
    % 1st output: The constraint corresponding to that labeling = (Groud Truth Feature - Worst Offending feature)
    % 2nd output: the margin you want to enforce for this constraint.
    % use global variable to communicate with @findMVC_BinaryLinearSVM without
    % passing the data, which can be very expensive.
    global patterns2train;
    global costs2train ;
    
    [~,yhat] = max(costs2train{id}+patterns2train{id}*w); % margin rescaling
    X_wo = (patterns2train{id}(1,:) -  patterns2train{id}(yhat,:))';
    margin = costs2train{id}(yhat);
end


