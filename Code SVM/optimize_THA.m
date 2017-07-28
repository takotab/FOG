function [th2_star, feval, result ] = optimize_THA(TrainY,SW,predicted_Y_train,timelimit)
if nargin == 0
   load('optimize_MBTM_save.mat') 
   SW = includeSW;
   %other.timelimit = 250;
end
x_ini = [60 0.45 0.55];
options = optimset('MaxFunEvals',timelimit);

if exist('SW','var') && ~isempty('SW') && size(SW,1) >3 %split because with sample weight takes longer
    functionincludingparameters = @(x_ini)minimizethisfunction_th_SW(x_ini,predicted_Y_train,TrainY,SW);
else
    functionincludingparameters = @(x_ini)minimizethisfunction_th(x_ini,predicted_Y_train,TrainY);
end
[th2_star,feval] = fminsearch(functionincludingparameters,x_ini,options);

PredictedYtest_w_th = thresholdsSVMAhlrichs(predicted_Y_train,th2_star,'forwards');
result = calcresults(PredictedYtest_w_th, TrainY);

end
function output = minimizethisfunction_th_SW(x_ini,outcome,Ytrain, sampleweight)

PredictedYtest = thresholdsSVMAhlrichs(outcome,x_ini,'forwards');
%[result] = calcresults_sampleweight(PredictedYtest, Ytrain, sampleweight);
[result] = calcresults(PredictedYtest,Ytrain, [], 0,0,sampleweight);
output = reward_function(result);

end

function output = minimizethisfunction_th(x_ini,outcome,Ytrain)
if x_ini(1) < 0 || x_ini(2)<0 || x_ini(3) < x_ini(2) || x_ini(3) >1
    output = nan;
else
    try
        PredictedYtest = thresholdsSVMAhlrichs(outcome,x_ini,'forwards');
        [result] = calcresults(PredictedYtest, Ytrain);
        output = reward_function(result);
    catch
       output = nan; 
    end
end
end

function [reward] = reward_function(result)
reward = -result.geomean * result.datausage;
if result.sensitivity < 0.7
    reward = reward * result.sensitivity;
end
if result.specificity < 0.7
    reward = reward * result.specificity;
end
end