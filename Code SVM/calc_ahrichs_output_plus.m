function [ r ] = calc_ahrichs_output_plus( features )
%calc_ahrichs_output cals the results of ahlirchs method out of the splistruct
if nargin == 0
    obj_ = load('daphnet_split_data_features.mat');
    obj_ = obj_.daphnet_obj;
    features = obj_.features;
    
end


addpath('C:\Users\Tako\Google Drive\Documenten\FOG\Research Internship Tako FOG Code with Andreas\Code SVM')
obj = classify_model(1);
table = zeros(9,3);
table_traing = nan(10,9);
table_test1 = nan(10,9);
table_test2 = nan(10,9);
for i = 1:10
    split_dataset = new_order_of_id(features,i);
   obj = classify_model(1);
    
    obj.mdl_setting.cachesize = 16000;
    obj.mdl_setting.featureset = 'ahlrichs';
    obj.mdl_setting.num_iterations = 1e6;
    
    [obj, r(i).result_training, predicted_Y_train] = obj.train(split_dataset);    
        
    [th1_star, th2_star] = calcoptimalsettings(predicted_Y_train,split_dataset.TrainY,1);
    
    [PredictedYtrain_th1] = thresholdsSVMAhlrichs(predicted_Y_train,th1_star);
    results_trainingth1 = calcresults(PredictedYtrain_th1,split_dataset.TrainY,'Results TH1');
    [PredictedYtrain_th2] = thresholdsSVMAhlrichs(predicted_Y_train,th1_star);
    r(i).results_trainingth2 = calcresults(PredictedYtrain_th2,split_dataset.TrainY,'Results TH2');
    table_traing(i,:) = [reshape(r(i).results_trainingth2.confmat',1,4),r(i).result_training.row];
    
    [~, predicted_Y_test] = obj.eval(split_dataset.TestX , split_dataset.TestY, 'Test result');
    
    % TH1
    [PredictedYtest_th1] = thresholdsSVMAhlrichs(predicted_Y_test,th1_star);
    r(i).results_testth1 = calcresults(PredictedYtest_th1,split_dataset.TestY,'Results TH1');
    table_test1(i,:) = [reshape(r(i).results_testth1.confmat',1,4),r(i).results_testth1.row];
    
    % TH2
    [PredictedYtest_th2] = thresholdsSVMAhlrichs(predicted_Y_test,th2_star);
    r(i).results_testth2 = calcresults(PredictedYtest_th2,split_dataset.TestY,'Results TH2');
     table_test2(i,:) = [reshape(r(i).results_testth2.confmat',1,4),r(i).results_testth2.row];
%     
%     table(:,1) = table(:,1) + 1/9 *  table_traing(i,:)';
%     table(:,2) = table(:,2) + 1/9 *  table_test1(i,:)';
%     table(:,3) = table(:,3) + 1/9 *  table_test2(i,:)';
    
end
table_traing(isnan(table_traing)) = 0;
table_test1(isnan(table_test1)) = 0;
table_test2(isnan(table_test2)) = 0;

table(:,1) = mean(table_traing(1:9,:),1);
table(:,2) = mean(table_test1(1:9,:),1);
table(:,3) = mean(table_test1(1:9,:),1);

table_sd(:,1) = std(table_traing(1:9,:),1);
table_sd(:,2) = std(table_test1(1:9,:),1);
table_sd(:,3) = std(table_test1(1:9,:),1);

end


function [splitstruct] = new_order_of_id(features,test_id)
idx_test_id = features(:,80)==test_id;
splitstruct.TrainX = features(idx_test_id == 0,2:77);
splitstruct.TrainY =  features(idx_test_id == 0,1);

splitstruct.TestX = features(idx_test_id == 1,2:77);
splitstruct.TestY =  features(idx_test_id == 1,1);

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
