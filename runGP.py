import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
from scipy.optimize import fsolve
from scipy.special import gamma,gammaincc
import scipy as sp
from theano import shared
import os

ndraws = 1000
ntune = ndraws
NUTS_KWARGS = {'target_accept': 0.99,'max_treedepth':30}
SEED = 4260026 # from random.org, for reproducibility

#compute mean and standard error of the net of R-D for a given column
def getMeanSD(df,col):
    MeanR = df[df[col].str.contains("Republican")]["weight"].sum()/df["weight"].sum()
    MeanD = df[df[col].str.contains("Democrat")]["weight"].sum()/df["weight"].sum()
    Mean = MeanR-MeanD
    Std  = np.sqrt((MeanR*(1.-MeanR)+MeanD*(1.-MeanD)+2.*MeanD*MeanR)/df["weight"].sum())

    if Std == 0.:
        Std = 1.
    return pd.Series({"Mean":Mean,"Std":Std})

#Specify a hierarchical normal prior with a non centered parameterization
def hierarchical_normal(name, shape, mu=0.,cs=1.,sigma=None):
    delta = pm.Normal('delta_{}'.format(name), 0., 1., shape=shape)
    if sigma is None:
        sigma = pm.HalfNormal('sigma_{}'.format(name), sd=cs)
    
    return pm.Deterministic(name, mu + delta * sigma)

#For estimating the gamma parameters for the GP hyperpriors
lower = 2.
upper = 7.
def getResid(x):
    a = x[0]
    b = x[1]
    rL = gammaincc(a,b/lower)-0.01
    rU = 1.-gammaincc(a,b/upper)-0.01
    return [rL,rU]

#Fit a GP as outcome = GP(age,birthyear)
def runGP2(df,race,col):
    if race != "All":
        df = df[df["race"]==race]
    df = df.groupby(["survey year","birthyear"]).apply(getMeanSD,col).reset_index()
    df["age"] = df["survey year"]-df["birthyear"]

    #This rescales age and birthyear to 0,10 so the covariates play well with the hyperprior
    minBY = df["birthyear"].min()
    maxBY = df["birthyear"].max()
    delBY = maxBY-minBY
    df["birthyear"] = 10.*(df["birthyear"]-minBY)/delBY

    minAGE = df["age"].min()
    maxAGE = df["age"].max()
    delAGE = maxAGE-minAGE
    df["age"] = 10.*(df["age"]-minAGE)/delAGE

    meanY = df["Mean"].mean()
    y     = df["Mean"].values - meanY 
    X     = df[["birthyear","age"]].values

    #Create a grid of values for plotting the results
    xx,yy  = np.meshgrid(np.linspace(0.,10.,30), np.linspace(0.,10.,30))
    Xnew   = np.vstack([xx.flatten(),yy.flatten()]).T
    byOut  = delBY*Xnew[:,0]/10. + minBY
    ageOut = delAGE*Xnew[:,1]/10. + minAGE
    with pm.Model() as model:
        #GP with an exponentiated quadratic covariance
        rho = pm.InverseGamma("rho", alpha=alpha0, beta=beta0,shape=2)
        cov = pm.gp.cov.ExpQuad(2, rho)
        gp  = pm.gp.Marginal(cov_func=cov)
        y_ = gp.marginal_likelihood("obs",X=X,noise=df["Std"].values,y=y)
    
        trace = pm.sample(draws=ndraws,tune=ntune,init="adapt_diag",chains=3, random_seed=SEED)
    
        #after fitting, get the predicted values on the plotting grid
        fcond = gp.conditional("fcond", Xnew=Xnew)
        pp_trace = pm.sample_ppc(trace,vars=[fcond])

    pp_trace["fcond"] += meanY
    columns = ["iter_{}".format(i) for i in range(ndraws)]
    ppdf = pd.DataFrame(pp_trace["fcond"].T,columns=columns)
    ppdf["Age"] = pd.Series(ageOut,index=ppdf.index)
    ppdf["Birthyear"] = pd.Series(byOut,index=ppdf.index)
    def getDifference(dfa):
        maxValidAge = 2016-dfa["Birthyear"].values[0]
        dfa = dfa[dfa["Age"]<=maxValidAge]

        dfOld   = dfa[dfa["Age"]>50]
        dfYoung = dfa[dfa["Age"]<30]
        meanOld   = dfOld[columns].mean(axis=0)
        meanYoung = dfYoung[columns].mean(axis=0)
        diff      = meanOld-meanYoung

        out = {}
        out["Mean Old"]   = np.mean(meanOld)
        out["Mean Young"] = np.mean(meanYoung)
        out["Mean Diff"]  = np.mean(diff)

        out["Low Old"]   = np.percentile(meanOld,10)
        out["Low Young"] = np.percentile(meanYoung,10)
        out["Low Diff"]  = np.percentile(diff,10)

        out["High Old"]   = np.percentile(meanOld,90)
        out["High Young"] = np.percentile(meanYoung,90)
        out["High Diff"]  = np.percentile(diff,90)
        return pd.Series(out)

    #Create dataframes with differences between the averages when age is over 50 and under 30
    ppdf = ppdf.groupby("Birthyear").apply(getDifference).reset_index()

    #Store mean and prediction intervals
    low, high = np.percentile(pp_trace['fcond'], [10, 90], axis=0)
    mean      = np.mean(pp_trace['fcond'], axis=0)
    dfOut = pd.DataFrame(np.vstack([byOut,ageOut,mean,low,high]).T,columns=["Birthyear","Age","Mean","Low","High"])
    dfOut["Race"] = race
    return dfOut,ppdf

#This fits a model as a[birthyear] + b[birthyear]*age where the coefficients a and b are drawn from a multilevel model, with group mean modeled by a GP
def runGPLin(df,race,col):
    if race != "All":
        df = df[df["race"]==race]
    df = df.groupby(["survey year","birthyear"]).apply(getMeanSD,col).reset_index()
    df["age"] = df["survey year"]-df["birthyear"]

    #rescale birthyear and center age at 50
    minBY = df["birthyear"].min()
    maxBY = df["birthyear"].max()
    delBY = maxBY-minBY
    birthyearIdx = df["birthyear"].values-minBY
    birthyearIdx = birthyearIdx.astype(np.int64)
    df["birthyear"] = 10.*(df["birthyear"]-minBY)/delBY

    nB = len(df["birthyear"].unique())
    age = (df["age"]-50.)/50.
    byUnq = np.unique(df["birthyear"])
    byOut  = delBY*byUnq/10. + minBY

    with pm.Model() as model:
        #Use gaussian process prior for the group mean of the intercepts
        rho_a = pm.InverseGamma("rho_a", alpha=alpha0, beta=beta0)
        alp_a = pm.HalfNormal("alpha_a", sd=1)
        cov_a = alp_a**2 * pm.gp.cov.ExpQuad(1, rho_a)
        gp_a = pm.gp.Latent(cov_func=cov_a)
        mu_alpha = gp_a.prior("f_a", X=byUnq[:,None])
        #Use gaussian process prior for the group mean of the slopes
        rho_b = pm.InverseGamma("rho_b", alpha=alpha0, beta=beta0)
        alp_b = pm.HalfNormal("alpha_b", sd=1)
        cov_b = alp_b**2 * pm.gp.cov.ExpQuad(1, rho_b)
        gp_b = pm.gp.Latent(cov_func=cov_b)
        mu_beta = gp_b.prior("f_b", X=byUnq[:,None])
    
        #multilevel model for the intercepts and slopes
        alpha = hierarchical_normal("alpha",mu=mu_alpha,shape=nB)
        beta  = hierarchical_normal("beta",mu=mu_beta,shape=nB)
    
        #and finally the observation model
        f = alpha[birthyearIdx]+beta[birthyearIdx]*age
        lk = pm.Normal("obs",mu=f,sd=df["Std"].values,observed=df["Mean"].values)
    
        trace = pm.sample(draws=ndraws,tune=ntune,init="adapt_diag",chains=3, random_seed=SEED,nuts_kwargs=NUTS_KWARGS)

    pm.traceplot(trace)
    plt.savefig("trace_{}_{}.png".format(col,race))
    #y*10 = alpha + beta * (age-50.)/50., where y is scaled on the order of 10s
    #so the factor of 10/50 destandardizes the coeff and converts to percentage
    mean = .2*np.mean(trace["f_b"],axis=0)
    low  = .2*np.percentile(trace["f_b"],10,axis=0)
    high = .2*np.percentile(trace["f_b"],90,axis=0)

    #Store the results in a df
    dfOut = pd.DataFrame(np.vstack([byOut,mean,low,high]).T,columns=["Birthyear","Mean","Low","High"])
    dfOut["Race"] = race
    return dfOut


#Estimate inverse gamma paramters for the gp length scale. See the following for details
#https://betanalpha.github.io/assets/case_studies/gp_part1/part1.html
x0 = [5.,5.]
alpha0,beta0 = fsolve(getResid,x0)
x = np.linspace(0.,10.,1000)
def IGammaPDF(x,alpha,beta):
    return beta**alpha/gamma(alpha)*x**(-alpha-1.)*np.exp(-beta/x)
plt.figure()
plt.plot(x,IGammaPDF(x,alpha0,beta0))
plt.savefig("Gammas.png")
plt.close()


#load the cleaned data
df = pd.read_csv(os.path.join("Data","anes_cleaned.csv"))


#run each model
dfGP2,dfGPDiff = runGP2(df.copy(),"All","party")
dfGP2.to_csv("gp2Party.csv",index=False)
dfGPDiff.to_csv("diffParty.csv",index=False)

dfGP2,dfGPDiff = runGP2(df.copy(),"All","presVote")
dfGP2.to_csv("gp2Pres.csv",index=False)
dfGPDiff.to_csv("diffPres.csv",index=False)

dfGP2,dfGPDiff = runGP2(df.copy(),"All","ideology")
dfGP2.to_csv("gp2Ideo.csv",index=False)
dfGPDiff.to_csv("diffIdeo.csv",index=False)

dfGPLin = runGPLin(df.copy(),"All","party")
dfGPLin.to_csv("gpLinParty.csv",index=False)

dfGPLin = runGPLin(df.copy(),"All","presVote")
dfGPLin.to_csv("gpLinPres.csv",index=False)

dfGPLin = runGPLin(df.copy(),"All","ideology")
dfGPLin.to_csv("gpLinIdeo.csv",index=False)
