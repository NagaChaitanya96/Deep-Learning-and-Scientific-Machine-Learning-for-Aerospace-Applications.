clc;
clear all;
close all;
dir

%%
data = load('mumbaielevation.csv');
X = data(:,1);
Y = data(:,2);
Z = data(:,3);
%%
% plot3(X,Y,Z)
surfl(data)