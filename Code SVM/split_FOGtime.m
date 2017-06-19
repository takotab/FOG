function SplitStruct = split_FOGtime(FOGtime,SplitStruct)
if size(FOGtime,2) == 3
   FOGtime(:,4) = 0; 
end

split_time = (SplitStruct.TrainX(end,79))+1.28; %because that is the middel so plus 256/2
idx_trainFOGtime = find(FOGtime(:,1)<split_time,1,'last');
FOGtime = [FOGtime(1:idx_trainFOGtime-1,:);...
    FOGtime(idx_trainFOGtime,1),split_time,FOGtime(idx_trainFOGtime,3:4);...
    split_time,FOGtime(idx_trainFOGtime,2),FOGtime(idx_trainFOGtime,3:4);...
    FOGtime(idx_trainFOGtime+1:end,:)];
    
SplitStruct.Train_FOGtime = FOGtime(1:idx_trainFOGtime,:);
SplitStruct.Test_FOGtime = FOGtime(idx_trainFOGtime+1:end,:);

idx_zero= find( SplitStruct.TestX(:,79)==0);
SplitStruct.TestX(idx_zero,:) = [];
SplitStruct.TestY(idx_zero,:) = [];
    
end
