function [result] = calcresults_sampleweight(outcome, true, sampleweight)
%%
if nargin == 0
    true = [ones(100,1);zeros(50,1)];
    outcome = [ones(90,1);zeros(60,1)];
    sampleweight = rand(150,1);
end
confusionmat_sampleweight(true, outcome, sampleweight);

result.datausage = nan;
result.confmat = [nan, nan;nan, nan];
result.sensitivity = nan;
result.specificity = nan;
result.accuracy = nan;
result.geomean = nan;
result.str = nan;
result.timeanddat = datetime('now');
%% calculate results
if ~isnan(true(1))
    try
                
        true(isnan(outcome)) = nan;
        
        result = calc_stats(true,outcome,sampleweight);
      
    catch
        warning('something in the results calculations did not work')
    end
end

end
%%
function result = calc_stats(true,outcome,sampleweight)

        result.datausage = 1-(sum(isnan(outcome))/length(outcome));
        
        result.confmat = confusionmat_sampleweight(true, outcome, sampleweight);
        result.sensitivity = result.confmat(2,2)/sum(result.confmat(2,:));
        result.specificity = result.confmat(1,1)/sum(result.confmat(1,:));
        result.accuracy = sum(diag(result.confmat))/sum(sum(result.confmat));
        result.geomean = sqrt(result.specificity * result.sensitivity );
        
        result.str = sprintf('%s: \n   sensitivity: \t%1.3f\n   specificity: \t%1.3f\n   data usage: \t\t%1.3f\n   geomean: \t\t%1.3f\n   accuracy: \t\t%1.3f\n   Confusion Matrix:\n\t\t\t\t\t%3.0f\t%3.0f\n\t\t\t\t\t%3.0f\t\t%3.0f\n',...
            'Sample weight results', result.sensitivity, result.specificity, result.datausage, result.geomean,result.accuracy ,result.confmat(1,1),result.confmat(1,2),result.confmat(2,1),result.confmat(2,2));
end
