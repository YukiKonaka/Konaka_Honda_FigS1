clear all
close all
clc

load("supple.mat")


for i=1:length(PS_C(:,1))
m_C(i)=mean(PS_C(i,1:end));
end

figure
hist(m_C,15)
xlim([-6,6])
xticks([-6 -3 0 3 6])
ylim([0,240])
yticks([0 60 120 180 240])
box off

hold on 
scatter(mean(ps_C),0,"r")

