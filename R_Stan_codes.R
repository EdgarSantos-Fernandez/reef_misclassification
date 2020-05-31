
.libPaths('C:\\1\\R')
library(Rcpp)
library(ggplot2)
library(dplyr)
library(RColorBrewer)
library(reshape2)
library(spdep)
library(rstan)
library(ggmcmc)
library(dismo)
library(sp)
library(rgeos)
library(geoR)
library(betareg)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
Sys.setenv(LOCAL_CPPFLAGS = '-march=corei7 -mtune=corei7')


m <- readRDS( 'm_20200601turk.RDS')
m_unique <- readRDS('m_unique_20200601turk.RDS')
perf_m_sum <- readRDS('perf_m_sum_20200601turk.RDS')
spatial_struc <- readRDS('spatial_struc.RDS')


model_beta <- betareg(yhat ~ dhw + notake + shelf + CYC -1, 
                      data = m)
summ <- summary(model_beta)$coefficients$mean


data_turk = list(
  N = nrow(m),                       # sample size
  M = length(unique(m$media_id)),    # num images
  Nindiv = length(unique(m$annot)),  # num participants
  id = m$media_id,                   # image id
  annot = m$annot,                   # participants id
  
  N_yhat_obs = length(which(!is.na(m$yhat))),
  N_yhat_mis = length(which(is.na(m$yhat))),
  ii_yhat_obs = which(!is.na(m$yhat)),
  ii_yhat_mis = which(is.na(m$yhat)),
  yhat_obs = m[!is.na(m$yhat), ]$yhat,
  
  N_y_obs = length(m_unique[!is.na(m_unique$y_truth2), ]$y_truth2) ,
  N_y_mis =  length(m_unique[is.na(m_unique$y_truth2), ]$y_truth2) ,
  ii_y_obs = which(!is.na(m_unique$y_truth2)),
  ii_y_mis = which(is.na(m_unique$y_truth2)),
  y_obs = m_unique[!is.na(m_unique$y_truth2), ]$y_truth2,
  
  K = 4,
  X = cbind(m$dhw, m$notake, m$shelf, m$CYC),
  
  beta1 = summ[1],
  beta2 = summ[2],
  beta3 = summ[3],
  beta4 = summ[4],
  
  alpha_se =  perf_m_sum$alpha_se ,
  beta_se =  perf_m_sum$beta_se ,
  
  alpha_sp = perf_m_sum$alpha_sp ,
  beta_sp = perf_m_sum$beta_sp,
  
  N_edges = spatial_struc$N_edges,
  node1 = spatial_struc$node1,
  node2 = spatial_struc$node2
)


model_turk <- '
data {
  int<lower=1> N;                   // sample size: num elictitation points
  int<lower=1> M;                   // num images
  int<lower=1> Nindiv;              // num of subjects
  int<lower=1> K;                   // K predictors
  int<lower = 1> id [N];            // image id
  int<lower=1,upper=Nindiv> annot[N]; // subjects id
  matrix[N,K] X;                    // design matrix
  int<lower = 0> N_yhat_obs; // number observed values
  int<lower = 0> N_yhat_mis; // number missing values
  int<lower = 1, upper = N_yhat_obs + N_yhat_mis> ii_yhat_obs[N_yhat_obs]; // ii index of observed
  int<lower = 1, upper = N_yhat_obs + N_yhat_mis> ii_yhat_mis[N_yhat_mis]; // ii index of missing
  real<lower=0,upper=1> yhat_obs[N_yhat_obs];
  int<lower = 0> N_y_obs; // number observed values
  int<lower = 0> N_y_mis; // number missing values
  int<lower = 1, upper = N_y_obs + N_y_mis> ii_y_obs[N_y_obs]; // ii index of observed
  int<lower = 1, upper = N_y_obs + N_y_mis> ii_y_mis[N_y_mis]; // ii index of missing
  real<lower=0,upper=1> y_obs[N_y_obs]; // true latent variable. 
  
  real beta1; 
  real beta2; 
  real beta3; 
  real beta4; 
  
  vector[Nindiv] alpha_se; // shape 1 for the se
  vector[Nindiv] beta_se;  // shape 2 for the se
  vector[Nindiv] alpha_sp; // shape 1 for the sp
  vector[Nindiv] beta_sp;  // shape 2 for the sp
  
  int<lower=0> N_edges;
  int<lower=1, upper=M> node1[N_edges];  
  int<lower=1, upper=M> node2[N_edges];  
}

parameters {
  vector[K] beta; 
  vector[M] phi0;
  real<lower = 0, upper = 1> alpha;
  real<lower=0,upper=1> yhat_mis[N_yhat_mis]; //declaring the missing yhat
  real<lower=0,upper=1> y_mis[N_y_mis]; //declaring the missing y
  real<lower=10, upper=200> phi; // dispersion parameter beta dist
  vector<lower=0.5, upper=1>[Nindiv] se; 
  vector<lower=0.5, upper=1>[Nindiv] sp; 
  real<lower=0> tau_phi; // precision of spatial effects
}

transformed parameters {
  vector[M] spat; // spatial prior
  vector<lower=0,upper=1>[M] mu;   // mean in the beta dist
  vector[M] Xbeta;                  // linear predictor
  real<lower=0,upper=1> yhat[N]; // creating yhat from the missing and observed values
  real<lower=0,upper=1> y[M]; // creating y from the missing and observed values
  real<lower=0> sigma_phi = inv(sqrt(tau_phi));  // convert precision to sigma 
  yhat[ii_yhat_obs] = yhat_obs;
  yhat[ii_yhat_mis] = yhat_mis;
  y[ii_y_obs] = y_obs; 
  y[ii_y_mis] = y_mis;
  spat = phi0 * sigma_phi;
  
  for (i in 1:N) { 
    Xbeta[id[i]] = X[i] * beta + spat[id[i]];   
    mu[id[i]] = inv_logit(Xbeta[id[i]]);   
    yhat[i] = se[annot[i]]  *  y[id[i]] + (1 - sp[annot[i]])  * (1 -  y[id[i]]) ;  
  }
}

model {
tau_phi ~ gamma(0.1, 0.1) ; 
target += -0.5 * dot_self(phi0[node1] - phi0[node2]); 
mean(phi0) ~ normal(0,0.001); 
target += beta_lpdf(y | (mu * phi), ((1.0 - mu) * phi));   //likelihood
//  priors
beta[1] ~ normal(beta1, 0.5);  
beta[2] ~ normal(beta2, 0.5);
beta[3] ~ normal(beta3, 0.5);
beta[4] ~ normal(beta4, 0.5);

phi ~ normal(20, 10)T[5,100]; 
for (i in 1 : Nindiv) {
  se[i] ~ beta(alpha_se[i], beta_se[i]);
  sp[i] ~ beta(alpha_sp[i], beta_sp[i]);
}
}
'

saveRDS(list(
  beta1 = summ[1,1], 
  beta2 = summ[2,1], 
  beta3 = summ[3,1], 
  beta4 = summ[4,1], 
  y = rep(0.50, length(unique(m_unique$media_id))), 
  yhat = rep(0.50, length(m$media_id))
), 'ini_turk_20200601.RDS')

initial <- readRDS( 'ini_turk_20200601.RDS')
ini <- function(){list(beta1 = initial$beta1, 
                       beta2 = initial$beta2, 
                       beta3 = initial$beta3, 
                       beta4 = initial$beta4, 
                       phi = list(10), 
                       y = initial$y, 
                       yhat = initial$yhat 
)}  

# NB: fit on a HPC node
m_turk <- stan(
  model_code = model_turk,
  model_name = "model_turk",
  data = data_turk,
  pars = c('beta', 'phi', 'se', 'sp', 'y', 'yhat', 'phi0', 'spat'),
  init = ini,
  iter = 24000, 
  warmup = 12000, 
  thin = 3,
  chains = 3,
  verbose = F,
  seed = 100,
  refresh = max(24000 / 100, 1) )

saveRDS(m_turk, paste0('model_', gsub(":", "", Sys.time()),'_','.rds'))
