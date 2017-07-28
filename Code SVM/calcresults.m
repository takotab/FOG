function [result] = calcresults(outcome,y_, name, other_eval_method,multiple_classes,x_count,sampleweight,ws,overlap,confmat)
if nargin == 0
   load('calc_all_results_save.mat') 
end
%%
if ~exist('matrin_eval','var')
    matrin_eval = 0;
end
if ~exist('multiple_classes','var')
    multiple_classes = 0;
end
if ~exist('x_count','var')
    x_count = 0;
end
if ~exist('sampleweight','var')
    sampleweight = pi .*ones(size(y_));
end
if ~exist('other_eval_method','var')
    other_eval_method = 0;
end

result.datausage = nan;
if exist('confmat','var')
    result.confmat = confmat;
    y_(1) = 0;
else
    result.confmat = [nan, nan;nan, nan];
end
result.sensitivity = nan;
result.specificity = nan;
result.accuracy = nan;
result.geomean = nan;
result.str = nan;
result.cost = nan;
result.matrix_pp_row = [nan,nan,nan];
result.timeanddat = datetime('now');
%% calculate results
if ~isnan(y_(1)) 
 %   try
        if nargin == 2
            name = 'Results';
        end
                
        y_(isnan(outcome)) = nan;
        result.datausage = 1-(sum(isnan(outcome))/length(outcome));
        if isnan(result.confmat(1,1))
            if multiple_classes == 1
                result.confmat = confusionmat(y_, outcome);
            elseif other_eval_method == 1     %bachlin
                result.confmat = x_countTxFx(y_==1, outcome==1,[],[]);
            elseif other_eval_method == 2     %martin
                result.confmat = eval_martinmethod(y_, outcome,ws,overlap);
            else
                if sampleweight(1) ~= pi
                    result.confmat = confusionmat_sampleweight(y_, outcome, sampleweight);
                else
                    result.confmat = confusionmat(y_==1, outcome==1);
                end
            end
        end
        result = calc_stats(result,name);
        
        if nargin==3 || other_eval_method ~=0
            disp(result.str)
        end
 %   catch e
%        warning(['something in the results calculations did not work' e])
%    end
end

end
%%
function result = calc_stats(result,name)


        if length(result.confmat) == 1 % only one class did occure
            result.confmat = [result.confmat, 0;0, 0];
        end
        result.accuracy = sum(diag(result.confmat))/sum(sum(result.confmat));
        result.sensitivity = result.confmat(2,2)/sum(result.confmat(2,:)); % TP/(TP+FN)
        result.specificity = result.confmat(1,1)/sum(result.confmat(1,:)); % TN/(TN+FP)
        result.geomean = sqrt(result.specificity * result.sensitivity );
        result.cost = cost_function(result);
        result.matrix_pp_row = [result.geomean,result.sensitivity,result.specificity,result.accuracy,result.cost];
        result.row = [result.accuracy,result.sensitivity,result.specificity,result.cost,result.geomean];
        result.row_me = [result.confmat(2,2), result.confmat(2,1), result.confmat(1,1), result.confmat(1,2), result.sensitivity,result.specificity,result.cost,result.geomean,result.accuracy];
        result.row_ahlrichs = [result.sensitivity,result.specificity,result.datausage,result.geomean,result.accuracy];
        result.str = sprintf('%s: \n   sensitivity: \t%1.3f\n   specificity: \t%1.3f\n   data usage: \t\t%1.3f\n   geomean: \t\t%1.3f\n   accuracy: \t\t%1.3f\n   Confusion Matrix:\n\t\t\t\t\t%3.0f\t%3.0f\n\t\t\t\t\t%3.0f\t\t%3.0f\n',...
            name, result.sensitivity, result.specificity, result.datausage, result.geomean,result.accuracy ,result.confmat(1,1),result.confmat(1,2),result.confmat(2,1),result.confmat(2,2));
end
%%

function confmat = confusionmat_sampleweight(true, outcome, sampleweight)
confmat = zeros(2);

%ignore NaN values in true and outcome
nanrows = isnan(true) | isnan(outcome);
if any(nanrows)
    true(nanrows) = [];
    outcome(nanrows) = [];
    sampleweight(nanrows) = [];
end

sampleweight = sampleweight./sum(sampleweight)*length(sampleweight);

for i = 1:length(true)
    confmat(true(i)+1,outcome(i)+1) = confmat(true(i)+1,outcome(i)+1)+sampleweight(i);
end

end