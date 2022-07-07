clear all
close all
clc

for i=1:1000

rng shuffle
sl1=0.9*ones(1,20);sl2=0.5*ones(1,41);sl3=0.5*ones(1,62);
sl4=0.1*ones(1,94);sl5=0.5*ones(1,20);sl6=0.9*ones(1,40);
sl7=0.1*ones(1,74);sl8=0.5*ones(1,40);
sl=[sl1,sl2,sl3,sl4,sl5,sl6,sl7,sl8];

sr1=0.5*ones(1,20);sr2=0.9*ones(1,41);sr3=0.1*ones(1,62);
sr4=0.5*ones(1,94);sr5=0.9*ones(1,20);sr6=0.5*ones(1,40);
sr7=0.5*ones(1,74);sr8=0.1*ones(1,40);
sr=[sr1,sr2,sr3,sr4,sr5,sr6,sr7,sr8];

Pl=sl;
Pr=sr;

T=length(sl);
time=1:T;

a=zeros(1,T);
o=zeros(1,T);
ml=zeros(1,T);
mr=zeros(1,T);
pl=zeros(1,T);
pr=zeros(1,T);
Gl=zeros(1,T);
Gr=zeros(1,T);
fl=zeros(1,T);
fr=zeros(1,T);

C=0*cos(2*pi*time/(T/2));

tt=1:T;

Po=0.5;
vw=0.4;
am=0.05;
po=0.1;
mo=0;

a(1)=rand<0.5;

ml(1)=mo; 
pl(1)=po;

mr(1)=mo;
pr(1)=po;

dt=1;

for t=2:T
   
    if t==2
        a(t)=rand<0.5;
    else
       al_prob=1/(1+(exp(-(Gr(t-1)-Gl(t-1)))));
       a(t)=rand<al_prob;
    end
    
    fl(t)=(1/(1+exp(-ml(t-1))));   
    fr(t)=(1/(1+exp(-mr(t-1)))); 
    
    if a(t)==1
        o(t) = rand < Pl(t);
        
        dmldt=((1/pl(t-1))+vw)*(o(t)-fl(t));
        ml(t)=ml(t-1)+am*dmldt*dt;
        Sl(t)=(1/(1+exp(-ml(t))));
        pl(t)=((1/vw*pl(t-1))/(pl(t-1)+1/vw))+(Sl(t)*(1-Sl(t)));
        
        mr(t)=mr(t-1);
        pr(t)=((1/vw*pr(t-1))/(pr(t-1)+1/vw));
    
    else
        o(t) = rand < Pr(t);
        
        dmrdt=((1/pr(t-1))+vw)*(o(t)-fr(t));
        mr(t)=mr(t-1)+am*dmrdt*dt;
        Sr(t)=(1/(1+exp(-mr(t))));
        pr(t)=((1/vw*pr(t-1))/(pr(t-1)+1/vw))+(Sr(t)*(1-Sr(t)));
        
        ml(t)=ml(t-1);
        pl(t)=((1/vw*pl(t-1))/(pl(t-1)+1/vw));
         
    end 
    lamdal(t)=1/(1+exp(-ml(t)));
    lamdar(t)=1/(1+exp(-mr(t)));
   
  %%%%%%%%%%%%%expected free energy%%%%%%%%%%%%%%
  
POAl(t)=lamdal(t)+0.5*lamdal(t)*(1-lamdal(t))*(1-2*lamdal(t))*((1/pl(t))+vw);  
Al(t)= -lamdal(t)*log(lamdal(t))-(1-lamdal(t))*log(1-lamdal(t));
Bl(t)=-0.5*(lamdal(t)*(1-lamdal(t))*(1+(1-2*lamdal(t))*(log(lamdal(t))-log(1-lamdal(t)))))*((1/pl(t))+vw);
Cl(t)=(1-POAl(t))*log(1-POAl(t))+POAl(t)*log(POAl(t));
Dl(t)=-POAl(t)*log(Po/(1-Po))-(1-POAl(t))*0;

POAr(t)=lamdar(t)+0.5*lamdar(t)*(1-lamdar(t))*(1-2*lamdar(t))*((1/pr(t))+2*vw);
Ar(t)= -lamdar(t)*log(lamdar(t))-(1-lamdar(t))*log(1-lamdar(t));
Br(t)=-0.5*(lamdar(t)*(1-lamdar(t))*(1+(1-2*lamdar(t))*(log(lamdar(t))-log(1-lamdar(t)))))*((1/pr(t))+vw);
Cr(t)=(1-POAr(t))*log(1-POAr(t))+POAr(t)*log(POAr(t));
Dr(t)=-POAr(t)*log(Po/(1-Po))-(1-POAr(t))*0;

Gl(t)=C(t)*(Al(t)+Bl(t)+Cl(t))+Dl(t);
Gr(t)=C(t)*(Ar(t)+Br(t)+Cr(t))+Dr(t);

Precisionl(t)=pl(t-1)/(fl(t)^2)./((1-fl(t))^2);
Precisionr(t)=pr(t-1)/(fr(t)^2)./((1-fr(t))^2); 

end

save('FE_action_reward_2.mat','ml','pl','mr','pr','C','a','o','lamdal','lamdar','Precisionl','Precisionr','al_prob')

rng shuffle

numberofparticles=100000;

C_noise=0.5;
load ('FE_action_reward_2.mat')
N_step=length(a);

%%
%run particle filter 

myPF = particleFilter(@state_15_self_org,@like_15_self_org);

am0=1;
vw0 = 0.4;
Co = -5;
POo=0.5;
init=[0;0;0.1;0.1;Co;am0;vw0;POo];


F1 = init(1); 
F2 = init(2);
F3 = init(3); 
F4 = init(4);
F5 = init(5);
F6 = init(6);
F7 = init(7);
F8 = init(8);

F = [F1,F2,F3,F4,F5,F6,F7,F8];


S1 = (0.01)^2;
S2 = (0.01)^2;
S3 = (0.1)^2;
S4 = (0.1)^2;
S5 = (0.001);
S6 = (0.5); 
S7 = (0.5);
S8 = (0.5);
S= diag([S1,S2,S3,S4,S5,S6,S7,S8]);

initialize(myPF,numberofparticles,F,S);
myPF.Particles(3:4,:)=gamrnd(10,0.001,[2,numberofparticles]);
myPF.Particles(5,:)=-20+40*rand(1,numberofparticles);
myPF.Particles(6,:)=0.04+0.02*rand(1,numberofparticles);
myPF.Particles(7,:)=0.2+0.5*rand(1,numberofparticles);
myPF.Particles(8,:)=0.3+0.4*rand(1,numberofparticles);


myPF.StateEstimationMethod = 'mean';

myPF.ResamplingMethod = 'systematic';

zEst=zeros(length(a),length(F));



for k=1:length(a)

    zEst(k,:) = correct(myPF,a(k),o(k));
 
    predict(myPF,a(k),o(k),C_noise);
    
    vv(:,:,k)=cov(transpose(myPF.Particles(1:5,:)));
        
end

v(:,:,:)=vv;


%%

PF_ml=zEst(:,1);
PF_mr=zEst(:,2);
PF_pl=zEst(:,3);
PF_pr=zEst(:,4);
PF_C =zEst(:,5);
PF_am =zEst(:,6);
PF_vw =zEst(:,7);
PF_Po =zEst(:,8);

pf_ml=PF_ml;
pf_mr=PF_mr;
pf_pl=PF_pl;
pf_pr=PF_pr;
pf_C =PF_C;
pf_am =PF_am;
pf_vw =PF_vw;
pf_Po =PF_Po;

pf_v =v;
% 
pf_rpl=1./(1+exp(-pf_ml));
pf_rpr=1./(1+exp(-pf_mr));
pf_precl=pf_pl./(pf_rpl.^2)./((1-pf_rpl).^2);
pf_precr=pf_pr./(pf_rpr.^2)./((1-pf_rpr).^2);

filter=[{pf_ml},{pf_mr},{pf_pl},{pf_pr},{pf_C},{pf_v},{pf_vw(end)},{pf_am(end)},{pf_Po(end)}];


smoother=PS_15(filter,a,o,C_noise);


ps=cell2mat(smoother(1));
ps_v=cell2mat(smoother(2));
ps_ml=transpose(ps(1,:));
ps_mr=transpose(ps(2,:));
ps_pl=transpose(ps(3,:));
ps_pr=transpose(ps(4,:));
ps_C =transpose(ps(5,:));

ps_rpl=1./(1+exp(-ps_ml));
ps_rpr=1./(1+exp(-ps_mr));
ps_precl=ps_pl./(ps_rpl.^2)./((1-ps_rpl).^2);
ps_precr=ps_pr./(ps_rpr.^2)./((1-ps_rpr).^2);


v_ml=ps_v(1,1,:);
V_ml=transpose(reshape(v_ml,1,length(a)));
v_mr=ps_v(2,2,:);
V_mr=transpose(reshape(v_mr,1,length(a)));
v_pl=ps_v(3,3,:);
V_pl=transpose(reshape(v_pl,1,length(a)));
v_pr=ps_v(4,4,:);
V_pr=transpose(reshape(v_pr,1,length(a)));
v_C=ps_v(5,5,:);
V_C=transpose(reshape(v_C,1,length(a)));

SD_rml=sqrt(V_ml.*(ps_rpl.^2).*(1-ps_rpl).^2);
SD_rmr=sqrt(V_mr.*(ps_rpr.^2).*(1-ps_rpr).^2);
SD_rpl=sqrt(V_pl./((ps_rpl.^4)./((1-ps_rpl).^4)));
SD_rpr=sqrt(V_pr./((ps_rpr.^4)./((1-ps_rpr).^4)));
SD_C=sqrt(V_C);



%% 好奇心の推定値の平均
PS_C(i,:)=ps_C;



end


 %%
 rng default
rng(3)

numberofparticles=100000;

load ('animal_data.mat')
N_step=length(a);


myPF = particleFilter(@state_15_self_org,@like_15_self_org);

%%%%%%%%%%%%%%　set initial value　%%%%%%%%%%%%%%%
am0=1;
vw0 = 0.4;
Co = -5;
POo=0.5
init=[0;0;0.1;0.1;Co;am0;vw0;POo];

F1 = init(1); 
F2 = init(2);
F3 = init(3); 
F4 = init(4);
F5 = init(5);
F6 = init(6);
F7 = init(7);
F8 = init(8);

F = [F1,F2,F3,F4,F5,F6,F7,F8];



S1 = (0.01)^2;
S2 = (0.01)^2;
S3 = (0.1)^2;
S4 = (0.1)^2;
S5 = (0.001);
S6 = (0.5); 
S7 = (0.5);
S8 = (0.5);
S= diag([S1,S2,S3,S4,S5,S6,S7,S8]);

initialize(myPF,numberofparticles,F,S);
myPF.Particles(3:4,:)=gamrnd(10,0.001,[2,numberofparticles]);
myPF.Particles(5,:)=-15+30*rand(1,numberofparticles);
myPF.Particles(6,:)=0.04+0.02*rand(1,numberofparticles);
myPF.Particles(7,:)=0.2+0.5*rand(1,numberofparticles);
myPF.Particles(8,:)=0.3+0.4*rand(1,numberofparticles);



myPF.StateEstimationMethod = 'mean';

myPF.ResamplingMethod = 'systematic';

zEst=zeros(length(a),length(F));



for k=1:length(a)

    zEst(k,:) = correct(myPF,a(k),o(k));
 
    predict(myPF,a(k),o(k),C_noise);
    

    vv(:,:,k)=cov(transpose(myPF.Particles(1:5,:)));
    
        
end

v(:,:,:)=vv;


%%

PF_ml=zEst(:,1);
PF_mr=zEst(:,2);
PF_pl=zEst(:,3);
PF_pr=zEst(:,4);
PF_C =zEst(:,5);
PF_am =zEst(:,6);
PF_vw =zEst(:,7);
PF_Po =zEst(:,8);

pf_ml=PF_ml;
pf_mr=PF_mr;
pf_pl=PF_pl;
pf_pr=PF_pr;
pf_C =PF_C;
pf_am =PF_am;
pf_vw =PF_vw;
pf_Po =PF_Po;

pf_v =v;
pf_rpl=1./(1+exp(-pf_ml));
pf_rpr=1./(1+exp(-pf_mr));
pf_precl=pf_pl./(pf_rpl.^2)./((1-pf_rpl).^2);
pf_precr=pf_pr./(pf_rpr.^2)./((1-pf_rpr).^2);


filter=[{pf_ml},{pf_mr},{pf_pl},{pf_pr},{pf_C},{pf_v},{pf_vw(end)},{pf_am(end)},{pf_Po(end)}];

% smoothing
smoother=PS_15(filter,a,o,C_noise)


ps=cell2mat(smoother(1));
ps_v=cell2mat(smoother(2));
ps_ml=transpose(ps(1,:));
ps_mr=transpose(ps(2,:));
ps_pl=transpose(ps(3,:));
ps_pr=transpose(ps(4,:));
ps_C =transpose(ps(5,:));

ps_rpl=1./(1+exp(-ps_ml));
ps_rpr=1./(1+exp(-ps_mr));
ps_precl=ps_pl./(ps_rpl.^2)./((1-ps_rpl).^2);
ps_precr=ps_pr./(ps_rpr.^2)./((1-ps_rpr).^2);

v_ml=ps_v(1,1,:);
V_ml=transpose(reshape(v_ml,1,length(a)));
v_mr=ps_v(2,2,:);
V_mr=transpose(reshape(v_mr,1,length(a)));
v_pl=ps_v(3,3,:);
V_pl=transpose(reshape(v_pl,1,length(a)));
v_pr=ps_v(4,4,:);
V_pr=transpose(reshape(v_pr,1,length(a)));
v_C=ps_v(5,5,:);
V_C=transpose(reshape(v_C,1,length(a)));


SD_rml=sqrt(V_ml.*(ps_rpl.^2).*(1-ps_rpl).^2);
SD_rmr=sqrt(V_mr.*(ps_rpr.^2).*(1-ps_rpr).^2);
SD_rpl=sqrt(V_pl./((ps_rpl.^4)./((1-ps_rpl).^4)));
SD_rpr=sqrt(V_pr./((ps_rpr.^4)./((1-ps_rpr).^4)));
SD_C=sqrt(V_C);



save("supple.mat")


%%
f = figure;
f.Position(3:4) = [1000 750];


N_step=length(a);

width=25;
for i=1:N_step-width
    ppl(i+width)=mean(a(i:i+width));
    
end

