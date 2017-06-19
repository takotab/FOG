function [ geometric_mean ] = calc_ahrichs_output( split_dataset )
%calc_ahrichs_output cals the results of ahlirchs method out of the splistruct
if nargin == 0
    load('daphnet_split_data.mat')
    split_dataset = split_dataset_d;
end

addpath('C:\Users\Tako\Google Drive\Documenten\FOG\Research Internship Tako FOG Code with Andreas\Code SVM')
obj = classify_model(1);

obj.mdl_setting.cachesize = 16000;
obj.mdl_setting.featureset = 'ahlrichs';
obj.mdl_setting.num_iterations = 1e6;

[obj, result_training, predicted_Y_train] = obj.train(split_dataset);
[th1_star, th2_star] = calcoptimalsettings(predicted_Y_train,split_dataset.TrainY,5);

[PredictedYtrain_th1] = thresholdsSVMAhlrichs(predicted_Y_train,th1_star);
results_trainingth1 = calcresults(PredictedYtrain_th1,split_dataset.TrainY,'Results TH1');
[PredictedYtrain_th2] = thresholdsSVMAhlrichs(predicted_Y_train,th1_star);
results_trainingth2 = calcresults(PredictedYtrain_th2,split_dataset.TrainY,'Results TH2');

[result_test, predicted_Y_test] = obj.eval(split_dataset.TestX , split_dataset.TestY, 'Test result');

% TH1
[PredictedYtest_th1] = thresholdsSVMAhlrichs(predicted_Y_test,th1_star);
results_testth1 = calcresults(PredictedYtest_th1,split_dataset.TestY,'Results TH1');

% TH2
[PredictedYtest_th2] = thresholdsSVMAhlrichs(predicted_Y_test,th2_star);
results_testth2 = calcresults(PredictedYtest_th2,split_dataset.TestY,'Results TH2');

table_ahlrichs = array2table([results_trainingth1.row',...
    results_testth1.row',...
    results_trainingth2.row',...
    results_testth2.row'],'VaribleNames',{});


end

% 
% Training Results TH2: 
%    sensitivity: 	0.439
%    specificity: 	0.714
%    data usage: 		1.000
%    geomean: 		0.560
%    accuracy: 		0.647
%    Confusion Matrix:
% 					125	 50
% 					 32		 25
% 
% Test Results TH2: 
%    sensitivity: 	0.444
%    specificity: 	0.700
%    data usage: 		0.864
%    geomean: 		0.558
%    accuracy: 		0.661
%    Confusion Matrix:
% 					 35	 15
% 					  5		  4
