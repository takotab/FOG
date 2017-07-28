clear all;close all;clc;
obj = classify_model(0,0,'all'); 
%first input == 0 because im not allowed to commit the code produced by Rispens et al.
obj.mdl_setting.zscore = 1;
obj.mdl_setting.cachesize = 1e6;
obj.mdl_setting.num_iterations = 1e9;
obj.sample_frequency = 60;
obj.mdl_setting.only_id_with_fog = 2;
obj.mdl_setting.th.do = 1;
%this data comes from the open dataset daphnet it downloaded at 4-5-2017
%from https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait
%Note: this is not the dataset I used during my intership.

load('daphnet_19062017.mat') %data
[obj, windowed_data] = obj.make_windows(data);
[obj, feature_matrix] = obj.make_features();
[obj,split_dataset, table_train, table_test] = obj.make_splitdata([0.7 0.3]);
[obj, results_training, predicted_Y_train] = obj.train();
[result, y_predicted] = obj.eval(obj.split_dataset.TestX , obj.split_dataset.TestY, 'Test result');

[result_other, y_predicted] = obj.eval_bachlin_and_martin(y_predicted,obj.split_dataset.TestY, split_dataset.Test_FOGtime);

table = [results_training.row_me';result.row_me'];





