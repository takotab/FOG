function [result_test, predicted_Y_test] = eval_MBFA(featurematrix,MBTM,true_y,name)
if nargin ==0 
    load('eval_MBTM_save.mat');
end
if nargin < 2
    MBTM.Freeze_index_threshold = 200;
    MBTM.Energie_threshold = 2;
end


predicted_Y_test = zeros(size(featurematrix,1),1);
%go over every datasamples and determ outcome of the MBTM algorithm

predicted_Y_test( featurematrix(:,1) > MBTM.Freeze_index_threshold & ...
    featurematrix(:,2) > MBTM.Energie_threshold) = 1 ;
if nargin == 3
    result_test = calcresults(predicted_Y_test,true_y);
else
    result_test = calcresults(predicted_Y_test,true_y,name);    
end
end