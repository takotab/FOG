function [spreadsheet_result] = add_more_type_of_eval_methods(spreadsheet_result)
%ADD_MORE_TYPE_OF_EVAL_METHODS will add to spreadsheet_results another 2
%fields: bachlin and martin. These are the evaluation methods descreaped by
%resp. Bachlin et al. 2010 and Rodríguez-Martín et al. 2017
if nargin == 0
    load('sample_results_abc_and_resultstructs.mat')
    spreadsheet_result = spreadsheet_result(1,1);
end

spreadsheet_result.result_bachlin = add_bachlin(spreadsheet_result);
%spreadsheet_result.result_martin = add_martin(spreadsheet_result);


end
function result = add_bachlin(spreadsheet_result)
%this function adds the bachlin method
[result] = calcresults(spreadsheet_result.predicted_Y,spreadsheet_result.true_y, [], 1);


end   
function result = add_martin(spreadsheet_result)
%this function adds the bachlin method
[result] = calcresults(spreadsheet_result.info.predicted_Y,spreadsheet_result.true_y_martin, [], 2);


end   