classdef classify_model
    %classify_model classifys raw data from windowing to features to
    % training to testing.
    %   input
    %           data: is n by 9 like this
    %               time, accl1, accl2, accl3, gyr1, gyr2, gyr3, FOGnFOG, id
    %
    % made by Tako Tabak
    %
    % last edit 29-06-2017 
    % make ensamler
    
    properties
        windowed_data = nan(1,256,9);
        features = nan(1,80);
        split_dataset = struct();
        mdl_setting = struct();
        windowlength_sec = 2.56;
        window_length_frame = nan;
        overlap_percentages = 0.25;
        sample_frequency = 100;
        verbrose_setting = struct();
        feature_settings = struct();
        variblenames = {};
        gait_characteristics = nan;
        Mdl_MBFA = struct();
        FOGtime = nan(1,3);
    end
    
    methods
        function obj = classify_model(include_gait_characteristics, start_run,featureset)
            if ~exist('featureset','var')
                featureset = 'all';
            end
            obj.variblenames = {'FOGnFOG','Accelerometer_bin1','Accelerometer_bin2','Accelerometer_bin3','Accelerometer_bin4','Accelerometer_bin5','Accelerometer_bin6','Accelerometer_bin7','Accelerometer_bin8','Accelerometer_bin9','Accelerometer_bin10','Accelerometer_bin11','Accelerometer_bin12','Accelerometer_bin13','Accelerometer_bin14','Accelerometer_bin15','Accelerometer_bin16',...
                'Gyroscoop_bin1','Gyroscoop_bin2','Gyroscoop_bin3','Gyroscoop_bin4','Gyroscoop_bin5','Gyroscoop_bin6','Gyroscoop_bin7','Gyroscoop_bin8','Gyroscoop_bin9','Gyroscoop_bin10','Gyroscoop_bin11','Gyroscoop_bin12','Gyroscoop_bin13','Gyroscoop_bin14','Gyroscoop_bin15','Gyroscoop_bin16',...
                'Mean_amplidute_freeze','Mean_amplidute_loco','std_amplidute_freeze','std_amplidute_loco','entropy_amplitude_freeze','entropy_amplitude_loco','peak_amplitude_freeze','peak_amplitude_loco','Frequency_of_peak_freeze','Frequency_of_peak_loco',...
                'Freeze_index','Energy_Freeze_and_Loco','Stepcandence','prev_Stepcandence','prev_prev_Stepcandence',...
                'Movement_Intensity_1','Movement_Intensity_2','Movement_Intensity_3','Stride_Time_1','Stride_Time_Varability',...
                'Harmonic_Ratio_1','Harmonic_Ratio_2','Harmonic_Ratio_3','Index_of_Harmonicity_1','Index_of_Harmonicity_2','Index_of_Harmonicity_3',...
                'Low_Percentage_07_1','Low_Percentage_07_2','Low_Percentage_07_3','Low_Percentage_07_4','Low_Percentage_14_1','Low_Percentage_14_2','Low_Percentage_14_3','Low_Percentage_14_4',...
                'Max_Lyapunov_exp_1','Max_Lyapunov_exp_2','Max_Lyapunov_exp_3','Norm_Lyapunov_exp_1','Norm_Lyapunov_exp_2','Norm_Lyapunov_exp_3',...
                'Movement_Intensity_1_256','Movement_Intensity_2_256','Movement_Intensity_3_256','Low_Percentage_14_4_256','class_if_more_class','time','id'};
            if nargin == 0 || include_gait_characteristics == 0
                obj.feature_settings.gait_characteristics = 0;
            else
                obj.feature_settings.gait_characteristics = 1;                
            end
            obj.window_length_frame = round(obj.windowlength_sec*obj.sample_frequency);
            obj.feature_settings.col.id = 80;
            obj.feature_settings.col.additional_info = obj.feature_settings.col.id-2:obj.feature_settings.col.id;
            obj.feature_settings.col.gait_characteristics = 49:73;
            
            obj.mdl_setting.get_old_model = 0;
            obj.mdl_setting.mdl_filename = '';
            obj.mdl_setting.only_id_with_fog = 1;
            obj.mdl_setting.general = 1;
            obj.mdl_setting.zscore = 1;            
            obj.mdl_setting.lda = 0;
            obj.mdl_setting.cachesize = 4;
            obj.mdl_setting.score_y.do = 1;
            obj.mdl_setting.num_iterations = 2;
            obj.mdl_setting.featureset = featureset;
            obj.mdl_setting.optimize_hyper_par = 0;
            obj.mdl_setting.do_th = 0;
            keySet =   {'accel', 'accel+gyro', 'ahlrichs', 'nogait','all','all+','selective','selective+','MBFA'};
            infoSet = {'first 16, only accel bins', 'first 32, accel bins and gyrobins', 'accel bins + 13 features', 'everything exept the gaitcharacteristics descripted by rispens et al.','all features, \nincl the interpolated gaitcharacteristics excl. not interpolated ones (73:76)','everything','only the ones that show an imporatnce above std','only the ones that show an importance above std \nexept that now the gaitfeatures are just over the window\n and not interpolated from 10 seconds','Freeze index and the energy level (44:45)'};
            numberSet = {1:16, 1:32,[1:16 33:45], 1:48,1:72,1:77,[1 2 3 4 5 6 7 11 12 13 14 15 18 30 33 34 35 36 37 38 40 50 66],...
                [1 2 3 4 5 6 7 11 12 13 14 15 18 30 33 34 35 36 37 38 40 75 76],[44:45]};
            optimal_hyper = {[2.2113 3.2067],[1.0575e3 929.0905],[2.3125 0.4875],[1 1],[1.1960 2.4417],[1 1],[41.4859 4.4986],[1 1],[1 1]};
            %optimal_hyper(1) = box constrained 
            %optimal_hyper(2) = kernel size
            obj.mdl_setting.featureinfo = containers.Map(keySet,infoSet);
            obj.mdl_setting.featurenumbers = containers.Map(keySet,numberSet);
            obj.mdl_setting.optimal_hyper = containers.Map(keySet,optimal_hyper);
            % options:
%             accel      first 16, only accel bins
%             accel+gyro first 32, accel bins and gyrobins
%             ahlrichs   accel bins + 13 features
%             nogait     everything exept the gaitcharacteristics
%                        descripted by rispens et al.
%             all        all features, incl the interpolated
%                        gaitcharacteristics excl. not interpolated ones
%                        (73:76)
%             all+       everything
%             selective  only the ones that show an imporatnce above std
%             selective+ only the ones that show an importance above std
%                        exept that now the gaitfeatures are just over the 
%                        window and not interpolated from 10 seconds
%             MBFA       Freeze index and the energy level (44:45)

            obj.verbrose_setting.wait_time = 10;
            obj.verbrose_setting.num_of_windows = 10000;
            
            addpath([ 'Estimate gait quality'])
            addpath([ 'Estimate gait quality' filesep 'LyapunovScripts' ])
            
            obj.mdl_setting.hyper_parameters = obj.mdl_setting.optimal_hyper(obj.mdl_setting.featureset);
            
            
            if nargin >1 && start_run == 1
                load('data.mat')
                fogtime = getridofsmallnonfogparts(fogtime);
                FOGnFOG = makeFOGnFOGtimeserie(fogtime,time);                
                %num_of_observations = 248721*1.03;
                %data = [time(1:num_of_observations)',accel(1:num_of_observations,:),gyro(1:num_of_observations,:),FOGnFOG(1:num_of_observations,:),id(1:num_of_observations,:)];
                data = [time',accel,gyro,FOGnFOG,id];
                [obj, windowed_data] = obj.make_windows(data);
                [obj, feature_matrix] = obj.make_features();
                [obj,split_dataset] = obj.make_splitdata();
                [obj, results_training, predicted_Y_train] = obj.train();
                [result, y_predicted] = obj.eval(obj.split_dataset.ValX , obj.split_dataset.ValY, ['Test result ']);% num2str(i_min_plus_both) '_' num2str(ten_fifty) ]);
            end

        end
        
        function [obj, windowed_data] = make_windows(obj,data)
            % windowing makes windows of the data and adds these to obj.windowed_data
            % this is a tensor of (num_windows,timestep_within_window,9)
            % please make sure data is like this:
            % size(data) --> (n,9):
            %   time, accl1, accl2, accl3, gyr1, gyr2, gyr3, FOGnFOG, id
        	%initiating      	 
            obj.FOGtime = makefogtime(data(:,1),data(:,8),data(:,9));
            if obj.mdl_setting.only_id_with_fog > 0
                idx_fog = obj.FOGtime(:,3) >0.7;
                id_w_fog = obj.FOGtime(idx_fog,4);
                unique_id_w_fog = unique(id_w_fog);
                data_w_fog = zeros(1,9);
                fogtime_w_fog = zeros(1,4);
                for i = 1:size(unique_id_w_fog,1)
                    
                    %check if this id has more or equal number of FOG episodes than obj.mdl_setting.only_id_with_fog
                    if sum(id_w_fog == unique_id_w_fog(i)) >= obj.mdl_setting.only_id_with_fog
                        add_data = data(data(:,9)== unique_id_w_fog(i),:);
                        add_data(:,1) = add_data(:,1) - add_data(1,1) + data_w_fog(end,1) + (add_data(end,1)-add_data(end-1,1));
                        data_w_fog = [data_w_fog;add_data];
                        add_fogtime = obj.FOGtime(obj.FOGtime(:,4)==unique_id_w_fog(i),:);
                        add_fogtime(:,1:2) = add_fogtime(:,1:2) - add_fogtime(1,1) + fogtime_w_fog(end,2);
                        fogtime_w_fog = [fogtime_w_fog;add_fogtime];
                    end
                end
                data = [];
                data_w_fog(1,:) = [];
                fogtime_w_fog(1,:) = [];
                data = data_w_fog;
                obj.FOGtime = fogtime_w_fog;
            end
            
            obj.window_length_frame = round(obj.windowlength_sec*obj.sample_frequency);
            windowed_data = nan(ceil(size(data,1)/(obj.window_length_frame-obj.window_length_frame*obj.overlap_percentages)),obj.window_length_frame,9);
            i_windowed_data = 1;
            i_current_frame = 1;
            
            tic
            while i_current_frame + obj.window_length_frame < size(data,1)
                
                % go over every frame in this windo untill it is bigger than
                % windowslength_frames
                windowed_data(i_windowed_data,1:round(obj.windowlength_sec*obj.sample_frequency),:) = data(i_current_frame :i_current_frame+round(obj.windowlength_sec*obj.sample_frequency)-1,:);
                                
                i_windowed_data = i_windowed_data +1;
                i_current_frame = i_current_frame + obj.window_length_frame - floor(obj.window_length_frame*obj.overlap_percentages);
                
                if toc > obj.verbrose_setting.wait_time && i_windowed_data/obj.verbrose_setting.num_of_windows == floor(i_windowed_data/obj.verbrose_setting.num_of_windows)
                    
                    fprintf('num of windows done/total: %i/%i\n',i_windowed_data,size(windowed_data,1))
                end
            end
            if size(windowed_data,1) > i_windowed_data %possibly the last one does not get filled depended on rounding so if thats the case delete it
                windowed_data(end,:,:) = [];
            end
            if size(obj.windowed_data ,1) == 1
                obj.windowed_data = windowed_data;
                
            else
                obj.windowed_data = [obj.windowed_data;windowed_data];
                obj.FOGtime = [obj.FOGtime;makefogtime(data(:,1),data(:,8))];
                warning('The feature to add stuff to the the current data in the obj is not working any more you are advised to abodone this route.')
            end
            
            fprintf('\nThis function is done making %i windows out of %i seconds of data\nThis is %.2f sec per window\n',size(windowed_data,1),round(data(end,1)),data(end,1)/size(windowed_data,1))
            
            if obj.feature_settings.gait_characteristics == 1 % include gait characteristics propsed by rispens et al.
                obj = obj.calc_gait_characteristics(data);
            end
        end
        
        function [obj, feature_matrix] = make_features(obj, data, windowed_data)
            % features makes windows of resp. (if applicible) windowed_data
            % and obj.windowed_data  and resp. adds or stores these to
            % obj.features (num_windows,feature_vector), feature_matrix is
            % FOGnFOG(1),features(2:52),class_if_multi_class(53),time(54),id(55)
            % if gait_characteristics == 1 then:
            % FOGnFOG(1),features(2:52),gait_characteristics(53:77),class_if_multi_class(78),time(79),id(80)
            
            if ~exist('windowed_data','var') && (isempty(obj.windowed_data) && exist('data','var'))
                [obj, windowed_data] = obj.make_windows(data);
            elseif ~isempty('obj.windowed_data') && ~exist('windowed_data','var')
                windowed_data = obj.windowed_data;
            end
            feature_matrix = zeros(size(windowed_data,1),80);
            
            %could probally be faster by using matrixces but it is really
            %clean this way
            tic
            for i = 1:size(feature_matrix,1)
                %% class
                [feature_matrix(i,1), feature_matrix(i,obj.feature_settings.col.id-2)] = isFOGornFOG(windowed_data(i,:,8));
                %% features
                % get the corresponding energy in the frequency bins
                %accelerometer
                [psd,f] = fft_tako(euclid_dist(reshape(windowed_data(i,:,2:4),obj.window_length_frame,3)),obj.sample_frequency);
                feature_matrix(i,2:17) = bins(psd,f);
                % other features as proposed by alhrichs et al. 2016
                feature_matrix(i,34:48) = alhrichsfeatures(psd,f);
                %gyroscoop
                [psd_g,f_g] = fft_tako(euclid_dist(reshape(windowed_data(i,:,5:7),obj.window_length_frame,3)),obj.sample_frequency);
                feature_matrix(i,18:33) = bins(psd_g,f_g);
                % some additional features pilot studies should to be benifital
                feature_matrix(i,74:76) = nanstd(reshape(windowed_data(i,:,2:4),obj.window_length_frame,3));
                feature_matrix(i,77) = sum(psd(f<1.4))/sum(psd(:));
                % other info
                feature_matrix(i,obj.feature_settings.col.id-1) = median(windowed_data(i,:,1)); % time
                feature_matrix(i,obj.feature_settings.col.id) = median(windowed_data(i,:,9)); % id
                if toc > obj.verbrose_setting.wait_time && i/obj.verbrose_setting.num_of_windows == floor(i/obj.verbrose_setting.num_of_windows)
                    
                    fprintf('num of windows done/total: %i/%i\n',i,size(feature_matrix,1))
                end
                
            end
            %step canedance from previous and previous previous 
            feature_matrix(1:end-1,47) = feature_matrix(2:end,46);
            feature_matrix(1:end-1,48) = feature_matrix(2:end,47);
            
            if obj.feature_settings.gait_characteristics == 1
                feature_matrix(:,obj.feature_settings.col.gait_characteristics) = obj.gait_characteristics(end-size(feature_matrix,1)+1:end,:);
            end
            feature_matrix(isnan(feature_matrix)) = 0;
            if size(obj.features,1) == 1
                obj.features =feature_matrix;
            else
                obj.features =[obj.features;feature_matrix];
            end
            
            if obj.mdl_setting.zscore ==1
                obj.features(isnan(obj.features)) = 0;
                obj.features(:,2:obj.feature_settings.col.id-4) = zscore(obj.features(:,2:obj.feature_settings.col.id-4));
            end
            fprintf('\nclass is done making %i windows of features\n',size(feature_matrix,1))
        end
        %%
        function [obj,split_dataset, table_train, table_test, table_val] = make_splitdata(obj, Ratio, sampleweigths, save_in_object)
            % splits the data in training, validation, and test and saves
            % it in obj.splitdata. Inaddition you can get the different
            % tables for the different datasets. If nargout == 3 you get
            % the all the datasets in one table, if nargout >3 then it will
            % be split in the different datasets.
            
            if ~exist('Ratio','var');Ratio = [0.6 0.2 0.2]; end
            if ~exist('sampleweigths','var');sampleweigths = [];end
            obj.features(isnan(obj.features)) = 0; 
%             obj.features = obj.features(randsample(290,290),:)
            split_dataset = splitdata([obj.features,ones(size(obj.features(:,end))), obj.features(:,end)], obj.features(:,1), Ratio, sampleweigths,[],obj.FOGtime);
            
            if ~exist('save_in_object','var') || save_in_object == 1
                obj.split_dataset = split_dataset;
            end
            if nargout >3
                
                table_train = array2table([split_dataset.TrainX(:,1:end-1)],'VariableNames',obj.variblenames);
                if length(Ratio) == 3
                    table_val = array2table([split_dataset.ValX(:,1:end-1)],'VariableNames',obj.variblenames);
                end
                table_test = array2table([split_dataset.TestX(:,1:end-1)],'VariableNames',obj.variblenames);
            elseif nargout == 3
                
                table_train = array2table([split_dataset.TrainY,split_dataset.TrainX;split_dataset.ValY,split_dataset.ValX;split_dataset.TestY,split_dataset.TestX],'VariableNames',obj.variblenames);
            end
            
        end
        %%
        function [obj, results_training, predicted_Y_train] = train(obj, split_dataset, cachesize, num_iterations, featureset)
            %train the current model with the settings as the proporties of
            %the current class. Saves the result as a struct
            %obj.mdl_setting and returns also the result and predicted Y
            if ~exist('split_dataset','var')
                split_dataset = obj.split_dataset;
            end
            
            if exist('cachesize','var')
                obj.mdl_setting.cachesize = cachesize;
            elseif ~isfield(obj.mdl_setting,'cachesize')
                obj.mdl_setting.cachesize = 16000;
            end
            if exist('num_iterations','var')
                obj.mdl_setting.num_iterations = num_iterations;
            elseif ~isfield(obj.mdl_setting,'num_iterations')
                obj.mdl_setting.num_iterations = 1e6;
            else
            end
            if exist('featureset','var')
                obj.mdl_setting.featureset = featureset;
            end
            
            % select featureset
            fprintf('Featureset %s chosen\tgood choise sur/madame;)\nThis is featureset is decripted as following:\n%s\n\nPlese type the following code to see your exect featureset:\nobj.variblenames(obj.mdl_setting.featurenumbers(obj.mdl_setting.featureset)+1)\n\nWe now you have a choise in featuresets\nand are gratefull that choise %s.\nHappy coding\n',obj.mdl_setting.featureset,obj.mdl_setting.featureinfo(obj.mdl_setting.featureset),obj.mdl_setting.featureset)
            trainX = split_dataset.TrainX(:,obj.mdl_setting.featurenumbers(obj.mdl_setting.featureset)+1);
            
            %change variblenames
            obj.variblenames = obj.variblenames(obj.mdl_setting.featurenumbers(obj.mdl_setting.featureset));
            
            if obj.mdl_setting.lda == 1 %LDA
                [l,v] = lda_t(trainX,split_dataset.TrainY);
                obj.mdl_setting.lda_v = real(v(:,1));
                trainX = trainX * obj.mdl_setting.lda_v;
                %change variblenames
                obj.variblenames = {'LDA_1'};
                
            end
            
            if obj.mdl_setting.zscore
               trainX = zscore(trainX); 
            end
            obj.mdl_setting.svm.cost = [ 0, sum(split_dataset.TrainY);sum(1-split_dataset.TrainY) 0];
            
            obj.mdl_setting.hyper_parameters = obj.mdl_setting.optimal_hyper(obj.mdl_setting.featureset);
            
            fprintf('\nFeatureset = %s\twith %i features\n',obj.mdl_setting.featureset,size(trainX,2))
            
            tic
            if obj.mdl_setting.general == 1 %general
                %%
                if obj.mdl_setting.optimize_hyper_par == 1
                    %% hyperparameter optimalization
                    fprintf('Start OptimizeHyperparameters\n')
                    rng default
                    if obj.mdl_setting.num_iterations ==1
                        fprintf('\nTaking the shortcut\n')
                        Mdl = fitcsvm(trainX,split_dataset.TrainY,'Cost',obj.mdl_setting.svm.cost,'CacheSize',obj.mdl_setting.cachesize,'Verbose',1,'IterationLimit',obj.mdl_setting.num_iterations,...
                            'BoxConstraint',0.027145,'KernelScale',1.3705);
                        
                    else
                        Mdl = fitcsvm(trainX,split_dataset.TrainY,'Cost',obj.mdl_setting.svm.cost,'ClassNames',[0 1],'OptimizeHyperparameters','auto',...
                        'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','ShowPlots',0));
                    end
                                        
                    ini = [Mdl.ModelParameters.BoxConstraint,Mdl.ModelParameters.KernelScale];
                    box_ker_hyper(1) = Mdl.ModelParameters.BoxConstraint ;
                    box_ker_hyper(1) = Mdl.ModelParameters.KernelScale ; %#ok<NASGU>
                    [f_eval, box_ker_star ] = optimize_svm(obj, trainX, split_dataset.TrainY,ini);
                    Mdl.ModelParameters.BoxConstraint = box_ker_star(1);
                    Mdl.ModelParameters.KernelScale = box_ker_star(2);
                    save(['hyperparametersearch_SVM_' date '_' obj.mdl_setting.mdl_filename '_obj.mat'],'Mdl','box_ker_star','f_eval','box_ker_hyper')
                    
                    
                    fprintf('\nWith new hyperparameters f_eval = %f',f_eval)
                    obj.mdl_setting.svm.Mdl = Mdl;
                else
                    %% normal
                    fprintf('Start SVM \nwith %i GB of cachesize and %i number of iterations\n', obj.mdl_setting.cachesize/1000, obj.mdl_setting.num_iterations);
                    
                    obj.mdl_setting.svm.Mdl = fitcsvm(trainX,split_dataset.TrainY,'Cost',obj.mdl_setting.svm.cost,'ClassNames',[0 1],...
                        'CacheSize',obj.mdl_setting.cachesize,'Verbose',1,'IterationLimit',obj.mdl_setting.num_iterations,...
                        'BoxConstraint',obj.mdl_setting.hyper_parameters(1),'KernelScale',obj.mdl_setting.hyper_parameters(2));
                    

                end
            else % individidual case
                
                % possible addition current data shows it is not worth the
                % computation time
            end
            %%
            fprintf('\nSVM training done in %i sec\n',toc)
            obj.mdl_setting.svm.toc = toc;
            [obj.mdl_setting.svm.predicted_Y_train, obj.mdl_setting.svm.predicted_Y_train_scorey] = predict(obj.mdl_setting.svm.Mdl,zscore(trainX));
            obj.mdl_setting.svm.results_training = calcresults(obj.mdl_setting.svm.predicted_Y_train,split_dataset.TrainY,'Result training SVM');
            
            if obj.mdl_setting.score_y.do == 1
                
                [ obj.mdl_setting.score_y.th, obj.mdl_setting.score_y.f_eval, obj.mdl_setting.score_y.results_training, obj.mdl_setting.score_y.predicted_Y_train] =...
                    optimize_scorey( obj.mdl_setting.svm.predicted_Y_train_scorey, split_dataset.TrainY, obj.mdl_setting.svm.predicted_Y_train);
                
                predicted_Y_train = obj.mdl_setting.score_y.predicted_Y_train;
                results_training = obj.mdl_setting.score_y.results_training;
                
                if obj.mdl_setting.th.do == 1
                    [obj.mdl_setting.th.th2star, obj.mdl_setting.th.f_eval, obj.mdl_setting.th.results_training] = optimize_THA(...
                        split_dataset.TrainY,[],obj.mdl_setting.score_y.predicted_Y_train,100);
                    
                    predicted_Y_train = obj.mdl_setting.score_y.predicted_Y_train;
                    results_training = obj.mdl_setting.score_y.results_training;
                end
            else
                
                predicted_Y_train = obj.mdl_setting.svm.predicted_Y_train;
                results_training = obj.mdl_setting.svm.results_training;
            end
            
        end
        %%
        function [obj] = fit_ensemble(obj, x, y)
           Mdl = fitcensemble(x,y,'CategoricalPredictors',[1 1 1 1],'CrossVal','on','ClassNames',{'nFOG','FOG'}) 
        end
        %%
        function [f_eval, box_ker_star ] = optimize_svm(obj, x, y, box_ker)
            if sum(y) >=2
                options = optimset('MaxFunEvals',30,'Display','notify');
                functionincludingparameters = @(box_ker)obj.minize_this_function(box_ker,x,y);
                [box_ker_star,f_eval] = fminsearch(functionincludingparameters,box_ker,options);
            else
                f_eval = 0;
                box_ker_star = box_ker;
            end
        end
        
        function [test_cost] = minize_this_function(obj,box_ker,x,y)
            if box_ker(1) <= 0 || box_ker(2) <= 0 
                test_cost = 1;
            else 
                idx_fog = find(y==1);
                n = size(x,1);
                n_train = round(n*0.8);
                confmat = [0 0;0 0];
                while sum(confmat(2,:)) < 5 || sum(confmat(1,:)) < 100 %|| confmat(2,2) < 1
                    idx = randsample(n,n,0);
                    Mdl = fitcsvm(x([idx(1:n_train);idx_fog(1)],:),y([1:n_train idx_fog(1)]),'Cost',obj.mdl_setting.svm.cost,'ClassNames',[0 1],...
                        'CacheSize',obj.mdl_setting.cachesize,...
                        'Verbose',0,'IterationLimit',obj.mdl_setting.num_iterations+1000,...
                        'BoxConstraint',box_ker(1),'KernelScale',box_ker(2));
                    y_predict = predict(Mdl,x([idx(n_train+1:end);idx_fog(randsample(length(idx_fog),1))],:));
                    confmat = confmat + confusionmat(y_predict==1,y([n_train+1:end idx_fog(end)])==1);                    
                end
                result = calcresults([],[], [], [],[],[],[],[],[],confmat);
                test_cost = result.cost;
            end
        end
        %%
        function [result, y_predicted] = eval(obj, test_dataX, test_dataY, name)
            % evaluates the current model as stored in properies of the
            % current class. returns the result and y_predicted
            if ~exist('name','var');name = 'Result test';end
            
            test_dataX = test_dataX(:,obj.mdl_setting.featurenumbers(obj.mdl_setting.featureset)+1);
            
            if obj.mdl_setting.lda == 1 %LDA
                
                test_dataX = test_dataX * obj.mdl_setting.lda_v;               
                
            end
            
            [y_predicted,y_predict_score] = predict(obj.mdl_setting.svm.Mdl, test_dataX);
            
            if obj.mdl_setting.score_y.do == 1
                
                [y_predicted] = eval_scorey(obj.mdl_setting.score_y.th, y_predict_score);
                
                 if obj.mdl_setting.th.do == 1
                     y_predicted = thresholdsSVMAhlrichs(y_predicted,obj.mdl_setting.th.th2star,'forwards');

                 end
            end
            result = calcresults(y_predicted,test_dataY,name);

        end
        %%
        function [result, y_predicted] = eval_bachlin_and_martin(obj, y_predicted,test_dataY, FOGtime, name)
            if ~exist('name','var');name = 'Result test';end
            result.bachlin = calcresults(y_predicted,test_dataY,[name ' bachlin'],1);
            result.martin = calcresults(y_predicted,FOGtime,[name ' martin'],2,[],[],[],obj.windowlength_sec,obj.windowlength_sec*obj.overlap_percentages);
            result.tabak = calcresults(y_predicted,FOGtime,[name ' tabak'],3,[],[],[],obj.window_length_frame,obj.overlap_percentages);
        end
        %%
        function obj = calc_gait_characteristics(obj, data)
            N10secEpochs = floor((size(data,1)* (1/obj.sample_frequency))/10);
            LocomotionMeasures(N10secEpochs).Measures = struct();
            x_times_to_intergrate = obj.windowed_data(:,128,1);
            fprintf('\nStart making gait characteristics of %i 10 sec windows\n',N10secEpochs)
            for epoch = 1:N10secEpochs
                [LocomotionMeasures(epoch).Measures] = GaitQualityFromTrunkAccelerations(data((epoch-1)*10*obj.sample_frequency+1:epoch*10*obj.sample_frequency,2:4),obj.sample_frequency,[0 0],0.00001,0);
                
                if epoch/obj.verbrose_setting.num_of_windows == floor(epoch/obj.verbrose_setting.num_of_windows)
                    
                    fprintf('num of 10sec epochs done/total: %i/%i\n',epoch,N10secEpochs)
                end
            end
            %% Gait features interpoleren
            Measure = [LocomotionMeasures.Measures];
            
            %%movement intensity
            MovementIntensity = reshape([Measure.StandardDeviation],[length(Measure),4]);
            GaitMeassure_10(:,1:3) = MovementIntensity(:,1:3);
            
            %stride time
            GaitMeassure_10(:,4) = [Measure.StrideTimeSeconds];
            
            %Stride time varability
            GaitMeassure_10(:,5) = [Measure.StrideLengthVariabilityOmitMinMax] ;
            
            %Harmonic Ratio
            HarmonicRatio = reshape([Measure.HarmonicRatio],[length(Measure),3]);
            GaitMeassure_10(:,6:8) = HarmonicRatio;
            
            %Index of Harmonicity
            IndexOfHarmonicity = reshape([Measure.IndexHarmonicity],[length(Measure),4]);
            GaitMeassure_10(:,9:11) = IndexOfHarmonicity(:,1:3);
            
            %Low HZ percentage
            LowHz = reshape([Measure.LowFrequentPercentage],[length(Measure),8]);
            GaitMeassure_10(:,12:19) = LowHz;
            
            %Max Lyapunov exp
            MaxLyapuexp = reshape([Measure.LyapunovW],[length(Measure),4]);
            GaitMeassure_10(:,20:22) = MaxLyapuexp(:,1:3);
            
            %Normalized Lyapanov exp
            normLyapunov = reshape([Measure.LyapunovPerStrideW],[length(Measure),4]);
            GaitMeassure_10(:,23:25) = normLyapunov(:,1:3);
            
            
            
            x10sec = 5:10:10*N10secEpochs-5;
            obj.gait_characteristics = nan(length(x_times_to_intergrate),25);
            for j = 1 : 25
                obj.gait_characteristics(:,j) = interp1(x10sec,GaitMeassure_10(:,j),x_times_to_intergrate,'pchip'); % interplate and also extrapolates to also the nan values
            end
            
        end
    end
    
end

