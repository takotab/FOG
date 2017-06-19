function fogtime = makefogtime(time,FOGnFOG,id)
%makes a table of 3/4 colunms with the first being the start the second the
%stop of the epidsode of the class in the third and if nargin ==3 its id 
%in the fourth colunm
if nargin == 0
   time = cumsum(ones(1,1000));
   FOGnFOG = [zeros(1,750),ones(1,100),zeros(1,150)];
end
if nargin <3
   id = ones(size(time)) ;
end
prev_class = 0;
i_fogtime = 1;
fogtime = [zeros(1,3),id(1)];
prev_id = id(1);

for i = 1:length(FOGnFOG)
    if FOGnFOG(i) ~= prev_class || id(i) ~= prev_id
        fogtime(i_fogtime,2) = time(i);
        i_fogtime = i_fogtime + 1;
        fogtime(i_fogtime,1) = time(i);        
        fogtime(i_fogtime,3) = FOGnFOG(i);
        fogtime(i_fogtime,4) = id(i);
        
    end
    prev_class = FOGnFOG(i);
    prev_id = id(i);
end
fogtime(end,2) = time(end);
if nargin ==2
   fogtime(:,4) = 0; 
end
end
