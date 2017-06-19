function [obj] = prep_obj(i_featureset,i_algorithm,obj)
switch i_featureset
    case 1
        obj.mdl_setting.featureset = 'accel';
    case 2
        obj.mdl_setting.featureset = 'accel+gyro';
    case 3
        obj.mdl_setting.featureset = 'ahlrichs';
    case 4
        obj.mdl_setting.featureset = 'all';
    case 5
        obj.mdl_setting.featureset = 'selective';
end

if i_algorithm == 2 %LDA
    obj.mdl_setting.lda =1;
end
end
