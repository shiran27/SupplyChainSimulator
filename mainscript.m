% clear all
% close all
% clc
% load('designedNetwork.mat');
% rng(7)
% 
% ACurl = []
% CCurl = []
% for i = 1:1:net.numOfChains
%     A = net.chains{i}.ACurl;
%     B = net.chains{i}.BCurl;
%     L = net.chains{i}.LLocal;
%     ABar = A+B*L;
%     BBar = eye(size(A));
%     CBar = net.chains{i}.CCurl;
%     sys1 = ss(ABar,BBar,CBar,0,-1);
%     figure
%     pzmap(sys1)
%     isstable(sys1)
%     ACurl = blkdiag(ACurl,ABar);
%     CCurl = blkdiag(CCurl,CBar);
% end
% BCurl = net.BCurl
% KCurl = net.KGlobal
% ACurl = ACurl + BCurl*KCurl*CCurl
% DCurl = net.DCurl
% sys2 = ss(ACurl,DCurl,CCurl,0,-1)
% pzmap(sys2)
% isstable(sys2)


clear all
close all
clc
load('designedNetwork.mat');
rng(7)


net.runSimulationAndSaveVideos('Case1',20)