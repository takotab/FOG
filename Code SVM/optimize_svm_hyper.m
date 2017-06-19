clear all;clc;
num_of_it = 1;
id_only_with_FOG = 1;
obj = classify_model(num_of_it~=1);
load('data14.mat')

[fogtime,cFOG] = getridofsmallnonfogparts(fogtime,0);
FOGnFOG = makeFOGnFOGtimeserie(fogtime,time);
num_of_observations = round(248721*2.03);

data = [time(1:num_of_observations)',accel(1:num_of_observations,:),gyro(1:num_of_observations,:),FOGnFOG(1:num_of_observations,:),id(1:num_of_observations,:)];
if num_of_it ~=1
    data = [time',accel,gyro,FOGnFOG,id];
end

obj.mdl_setting.only_id_with_fog = id_only_with_FOG;
obj.mdl_setting.optimize_hyper_par = 1;
obj.mdl_setting.num_iterations = num_of_it;

[obj, windowed_data] = obj.make_windows(data);
[obj, feature_matrix] = obj.make_features();
[obj, split_dataset, table_train_rr, table_test_rr] = obj.make_splitdata([0.8 0.2]);

tic
time = nan(5,1);
for i = 1:5
    parobj = obj;
    parobj.mdl_setting.mdl_filename = num2str(i);
    [parobj] = prep_obj(i,1,parobj);
    [parfor_obj ] = parobj.train(split_dataset);
    time(i,1) = toc;
    fprintf('\nCurrent i = %i\nElapsed time = %i sec\n',i,toc);
end



