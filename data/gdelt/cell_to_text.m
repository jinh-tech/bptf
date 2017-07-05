clc;
clear;
load gdelt.mat;
fileID = fopen('non_zero_indices.txt','w');
filexi = fopen('non_zero_values.txt','w');
fileactor = fopen('countries','w');
filedates = fopen('dates','w');
formatSpec = '%d %d %d %d\n';
for j = 1:1030625
    fprintf(fileID,formatSpec,id{1}(j),id{2}(j),id{3}(j),id{4}(j));
    fprintf(filexi,'%d\n',xi(j));
end

for j=1:220
    fprintf(fileactor,'%s\n',actors(j,1:3));
end

for j=1:53
    fprintf(filedates,'%s\n',dates(j,1:10));
end

fclose(fileID);
fclose(filexi);
fclose(fileactor);
fclose(filedates);