function smoother=PS_15(filter,a,o,C_noise)

vw=cell2mat(filter(7));

N=length(a);

pw=1/vw;

% m_noise=0;
% p_noise=0;

m_noise=0.1;
p_noise=0.1;
%C_noiseはPFの関数でstate_transition とともに一括制御

rr=[m_noise;m_noise;p_noise;p_noise;C_noise];
r=diag(rr);

%r=zeros(5,5);

pf_ml=transpose(cell2mat(filter(1)));
pf_mr=transpose(cell2mat(filter(2)));
pf_pl=transpose(cell2mat(filter(3))); 
pf_pr=transpose(cell2mat(filter(4)));
pf_C =transpose(cell2mat(filter(5)));
vw   =transpose(cell2mat(filter(7)));
am   =transpose(cell2mat(filter(8)));



m=[pf_ml;pf_mr;pf_pl;pf_pr;pf_C];
v=cell2mat(filter(6));

%スムーザーの平均Mと分散Vの初期値の設定(初期値はフィルターの最後の値)
M=zeros(5,N);
M(:,N)=m(:,N);
V(:,:,N)=v(:,:,N);

for t=1:N-1

Sl=1./(1+exp(-pf_ml(N-t)));
Sr=1./(1+exp(-pf_mr(N-t)));

%ml〜Cで異なる部分(Fの部分)を先に作成しておく。
%行列Fの作成

F(1,:)=pf_ml(N-t)+am*(vw+pf_pl(N-t)^-1)*(o(N-t+1)-Sl)*(a(N-t)==1);
F(2,:)=pf_mr(N-t)+am*(vw+pf_pr(N-t)^-1)*(o(N-t+1)-Sr)*(a(N-t)==0);

%もとバージョン
SF(1,:)=1/(1+exp(-F(1,:)+r(1,1)*randn)); 
SF(2,:)=1/(1+exp(-F(2,:)+r(2,2)*randn));

F(3,:)=((pw*pf_pl(N-t))/(pf_pl(N-t)+pw))+SF(1,:)*(1-SF(1,:));
F(4,:)=((pw*pf_pr(N-t))/(pf_pr(N-t)+pw))+SF(2,:)*(1-SF(2,:));
F(5,:)=0;


%行列F’の作成 (Fの一階微分)
FF(1,:)=1-am*(vw+pf_pl(N-t)^-1)*Sl*(1-Sl)*(a(N-t)==1);
FF(2,:)=am*(-(o(N-t+1)-Sl)/(pf_pl(N-t))^2)*(a(N-t)==1);%以前つけ忘れてたのでa==つけた
FF(3,:)=1-am*(vw+pf_pr(N-t)^-1)*Sr*(1-Sr)*(a(N-t)==0);
FF(4,:)=am*(-(o(N-t+1)-Sr)/(pf_pr(N-t))^2)*(a(N-t)==0);%以前つけ忘れてたのでa==つけた
FF(5,:)=FF(1,:)*SF(1,:)*(1-SF(1,:))*(1-2*SF(1,:));
FF(6,:)=(pw^2)/((pf_pl(N-t)+pw)^2);
FF(7,:)=FF(3,:)*SF(2,:)*(1-SF(2,:))*(1-2*SF(2,:));
FF(8,:)=(pw^2)/((pf_pr(N-t)+pw)^2);

%Aの作成
A=zeros(5,5);
A(1,1)=FF(1,:);A(1,3)=FF(2,:);A(2,2)=FF(3,:);A(2,4)=FF(4,:);
A(3,1)=FF(5,:);A(3,3)=FF(6,:);A(4,2)=FF(7,:);A(4,4)=FF(8,:);
A(5,5)=1;

%dの作成 
mm=m(:,N-t);
mm(5)=0; %失敗の原因ではない
d=F-A*mm;%+rr; 先生にいわれてrr除くと、良い兆しがみえた

%Pの作成
P=r+A*v(:,:,N-t)*transpose(A); %5x5 OK

%Jの作成
J=v(:,:,N-t)*transpose(A)*inv(P);  %5x5　OK

%スムーザーの更新
M(:,N-t)=m(:,N-t)+J*(M(:,N-t+1)-d-A*m(:,N-t)); %　OK
V(:,:,N-t)=v(:,:,N-t)+J*(V(:,:,N-t+1)-P)*transpose(J);

end
smoother=[{M} {V}];
