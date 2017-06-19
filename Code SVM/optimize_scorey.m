function [ score_y_th, f_eval, results_training, outcome] = optimize_scorey( score_y, true_y, predicted_y_matlab)
%optimize_scorey returns the best threshold to get the best geomean
%   Detailed explanation goes here
if nargin == 0
    load('optimize_scorey_save.mat')
end
if size(score_y ,2) >1
    score_y = score_y(:,2);
end

x_ini = 0;
functionincludingparameters = @(x_ini)minimizethisfunction(x_ini,score_y(:,1),true_y);

[score_y_th,f_eval.output_me] = fminsearch(functionincludingparameters,x_ini);

if nargout > 1 || nargin == 0
    
    [f_eval.result_matlab] = calcresults(predicted_y_matlab, true_y);
    f_eval.output_me = minimizethisfunction(score_y_th,score_y(:,1),true_y);
    if f_eval.result_matlab.cost > f_eval.output_me
        fprintf('matlab result:\t %f\nmy result:\t %f\n',f_eval.result_matlab.cost,f_eval.output_me)
    end
    
end

if nargout == 4
    outcome = eval_scorey(score_y_th, score_y(:,1));
    results_training = calcresults(outcome, true_y);
end

end
function output = minimizethisfunction(x_ini,scorey,Ytrain,name)
outcome = eval_scorey(x_ini, scorey);
if nargin < 4
    [result] = calcresults(outcome, Ytrain);
else
    [result] = calcresults(outcome, Ytrain, name);
end
output = result.cost;
end




