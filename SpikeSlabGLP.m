function [store_B,store_z,store_phi,store_q,store_R2,store_gam,store_s2,y,x,u,T,k]=SpikeSlabGLP(y,x,u,abeta,bbeta,Abeta,Bbeta,M,N)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Linear regression with a spike-and-slab prior, as in 
%   "Economic Predictions with Big Data: the Illusion of Sparsity,"
%   by Domenico Giannone, Michele Lenza and Giorgio Primiceri.
%
% Inputs:
%   y: T x 1 vector of observations of the response variable
%   x: T x k matrix of observations of the predictors 
%          (they should all have mean zero and the same std, say 1)
%
% Optional inputs 
% (use '[]' for each input you wish to set to its default value):
%   u: T x l matrix of observations of the predictors always included
%            (use this option to introduce a constant term in the regression)
%   abeta and bbeta: parameters of beta prior on q (default values: 1 and 1)
%   Abeta and Bbeta: parameters of beta prior on R2 (default values: 1 and 1)
%   M and N: total number of draws, and number of initial draws to be discarded
%
% Output:
%   store_B: posterior draws of regression coefficients
%   store_z: draws of 0-1 variables indicating if a predictor is included
%   store_phi: draws of regression coefficients of predictors always included
%   store_q: draws of probability of inclusion
%   store_R2: draws of R squared
%   store_gam: draws of prior standard deviation conditional on inclusion
%   store_s2: draws of residual variance
%   y: T x 1 vector of observations of the response variable
%   x: T x k matrix of observations of the predictors 
%   u: T x l matrix of observations of the predictors always included
%   T: number of observations
%   k: number of predictors
%
% Reference:
%   Giannone, Domenico, Michele Lenza and Giorgio E. Primiceri (2021) 
%   "Economic Predictions with Big Data: the Illusion of Sparsity,"
%   Econometrica, forthcoming.
%
% This code and the paper are available at
% https://faculty.wcas.northwestern.edu/~gep575/
%
% Last modified: March 9, 2021.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
if isempty(u); l=0; else; l=size(u,2); end;
if isempty(abeta) || isempty(bbeta); abeta=1; bbeta=1; end
if isempty(Abeta) || isempty(Bbeta); Abeta=1; Bbeta=1; end
if isempty(M); M=110000; N=10000; end
if isempty(N) && ~isempty(M); N=round(M/11); end


[T,k]=size(x);
varx=var(x(:,1),1);

%% edges of grids for q and R2
edgesq=[0:.001:.1 .11:.01:.9 .901:.001:1];
edgesR2=[0:.001:.1 .11:.01:.9 .901:.001:1];

% storage and preliminary calculations
if l>0; store_phi=zeros(l,M); end
store_B=zeros(k,M);
store_s2=zeros(M,1);
store_gam=zeros(M,1);
store_z=zeros(k,M);
store_q=zeros(M,1);
store_R2=zeros(M,1);

areaq=edgesq(2:end)-edgesq(1:end-1);
areaR2=edgesR2(2:end)-edgesR2(1:end-1);
areaqR2=repmat(areaq,length(areaR2),1).*repmat(areaR2',1,length(areaq)); % areaq.*areaR2';
intq=edgesq(1:end-1)+areaq/2;
intR2=edgesR2(1:end-1)+areaR2/2;
INTQ=repmat(intq,length(intR2),1);
INTR2=repmat(intR2',1,length(intq));

xx=x'*x; xy=x'*y; yx=y'*x; yy=y'*y;
if l>0; xu=x'*u; yu=y'*u; uu=u'*u; invuu=uu\eye(l); cholinvuu=chol(invuu); ux=u'*x; invuuuy=invuu*u'*y; invuuux=invuu*u'*x;
elseif l==0; xu=zeros(k,1); yu=zeros(1,1); uu=zeros(1,1); ux=zeros(1,k); phi=0; end
    
QR2=INTQ.*(1-INTR2)./INTR2;
for ttt=0:k
    prodINT(:,:,ttt+1)=INTQ.^(ttt+ttt/2+abeta-1).*(1-INTQ).^(k-ttt+bbeta-1).*...
        INTR2.^(Abeta-1-ttt/2).*(1-INTR2).^(ttt/2+Bbeta-1).*areaqR2;
end

% starting values (based on Lasso under the assumption that the x's explain 50% of var(resy))
if l>0; phi0=invuuuy; resy=y-u*phi0; store_phi(:,1)=phi0; else; resy=y; end
b0=lasso(x,resy,'Lambda',sqrt(8*k*varx*var(resy,1)/2)/(2*length(resy)),'Standardize',0);
store_B(:,1)=b0; b=b0;
s2=sum((resy-x*b).^2)/T; store_s2(1)=s2;
z=(b~=0); store_z(:,1)=z;
tau=sum(z);

% Gibbs sampling
tic
for i=2:M
    if i==1000*floor(.001*i);
        i
    end
    
    % draw q and R2
    pr_qR2=exp(-(.5*k*varx*b'*b/s2).*QR2).*squeeze(prodINT(:,:,tau+1));    
    cdf_qR2=cumsum(pr_qR2(:)/sum(pr_qR2(:)));
    aux=sum(cdf_qR2<rand(1))+1;
    
    q=INTQ(aux);
    store_q(i)=q;
    R2=INTR2(aux);
    store_R2(i)=R2;
    gam=sqrt((1/(k*varx*q))*R2/(1-R2));
    store_gam(i)=gam;
    
    % draw phi
    if l>0;
        phihat=invuuuy-invuuux*store_B(:,i-1);
        Vphi=invuu*s2;
        phi=(randn(1,l)*sqrt(s2)*cholinvuu)'+phihat;
        store_phi(:,i)=phi';
    end
    
    % draw z
    for j=1:k
        z0=z; z0(j)=0;
        z1=z; z1(j)=1;
        tau0=sum(z0);
        tau1=sum(z1);
        W0=xx(logical(z0),logical(z0))+eye(tau0)/gam^2;
        W1=xx(logical(z1),logical(z1))+eye(tau1)/gam^2;
        bhat0=W0\(xy(logical(z0))-xu(logical(z0),:)*phi);
        bhat1=W1\(xy(logical(z1))-xu(logical(z1),:)*phi);
        
        % note: 2*sum(log(diag(chol(W0)))) improves speed and accuracy of log(det(W0))
        log_pr_z0=tau0*log(q)+(k-tau0)*log(1-q)-tau0*log(gam)-.5*2*sum(log(diag(chol(W0))))...
            -.5*T*log(yy-2*yu*phi+phi'*uu*phi-yx(logical(z0))*bhat0+phi'*ux(:,logical(z0))*bhat0);
        log_pr_z1=tau1*log(q)+(k-tau1)*log(1-q)-tau1*log(gam)-.5*2*sum(log(diag(chol(W1))))...
            -.5*T*log(yy-2*yu*phi+phi'*uu*phi-yx(logical(z1))*bhat1+phi'*ux(:,logical(z1))*bhat1);
        
        z(j)=(rand(1)<=(1/(exp(log_pr_z0-log_pr_z1)+1)));
    end
    tau=sum(z);    
    W=xx(logical(z),logical(z))+eye(tau)/gam^2;
    bhat=W\(xy(logical(z))-xu(logical(z),:)*phi);
    store_z(:,i)=z;
    
    % draw s2
    niu = yy-2*yu*phi+phi'*uu*phi-bhat'*W*bhat;
    s2=1./gamrnd(T/2,2/niu);
    store_s2(i)=s2;
    
    % draw b
    b=(randn(1,tau)*chol(s2*eye(tau)/W))'+bhat;
    store_B(logical(z),i)=b;    
end
timeGibbs=toc


% plotting some results
figure('Position', [0, 400, 1000, 400]);

% posterior of the overall robability of inclusion (q)
subplot(1,2,1)
edges=0:.01:1; histogram(store_q(N+1:end),edges,'Normalization','pdf');
xlabel('q')
title('Posterior of q')
xlim([0 1])

% probability of inclusion of each predictor
ax1=subplot(1,2,2);
meanZ=mean(store_z(:,N+1:M)');
imagesc(meanZ)
colorbar
caxis([0, 1])
colormap(ax1,flipud(colormap(ax1,'hot')))
set(gca,'YTickLabel','','Ytick',[])
xlabel('coefficients')
title('Probability of inclusion of each predictor')