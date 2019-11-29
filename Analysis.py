## Imports
import pandas as pd
import numpy as np
from scipy import stats
import pickle as pkl
from sklearn import linear_model
import statsmodels.api as sm
import xlrd, os

## Move to working directory
os.chdir('C:/Users/Archie Wiranata/Documents/University/Semester 8/ECON4200')
pd.options.mode.chained_assignment = None

start = "30-11-17"
inclusion = "31-05-18" ## Effective at June 1
end = "12-01-18"


## Read SSE's data
def read_SSE():
    df = pd.read_excel("db/SSE.xlsx", skiprows=0, sheet_name=0)
    df = df[df["Date"]>"2017-11-30"]
    df = df[df["Date"]<"2018-12-01"]
    return df

SSE = read_SSE()
SSEBA = pd.read_csv('db/sse180BA.csv')
SSEBA['Date'] = pd.to_datetime(SSEBA['Date'], format='%Y-%m-%d')

'''
def read_SSE_components(name, skip):
    df = pd.read_excel(file, sheet_name=name, skiprows=skip)
    df.columns = ['Date', 'Price', 'Bid', 'Ask', 'Market cap', 'Cap']

    df['Bid'][df['Bid'].isnull()] = df['Price'][df['Bid'].isnull()]
    df['Ask'][df['Ask'].isnull()] = df['Price'][df['Ask'].isnull()]

    df['bidask'] = df['Bid'] - df['Ask']
    df = df[['Date','bidask', 'Cap']]
    df['Name'] = name

    if df.isnull().values.any():
        return None
    elif (len(df)!= 245):
        return None
    else:
        return df
## Initialize memory
dfs = []

file = 'db/Nat SSE 180.xlsx'
names = xlrd.open_workbook(file).sheet_names()
refer = names.pop(0)

for name in names:
    print (name)
    dfs.append(read_SSE_components(name, 2))

file = 'db/Anthony SSE Data.xlsx'
names = xlrd.open_workbook(file).sheet_names()
refer = names.pop(0)

for name in names:
    print (name)
    dfs.append(read_SSE_components(name, 1))

dfs = [df for df in dfs if  df is not None]
sse180 = pd.concat(dfs, ignore_index=True, sort='Date')

dates = sse180['Date'].unique()
sseBAs = []
for date in dates:
    df = sse180[sse180['Date'] == date]
    tot = sum(df['Cap'])
    w = df['Cap'] / tot
    sseBA = np.multiply(w,df['bidask'])
    sseBAs.append(sum(sseBA)/len(sseBA))

sse180 = pd.DataFrame({'Date':dates, 'bidask':sseBAs})
sse180.to_csv('db/sse180BA.csv', index=False)

def read_in(name, skips):
    df = pd.read_excel(file, sheet_name=name, skiprows=skips)
    df = df[1:]

    if (len(df['Daily price'])>sum(df['Daily price'].isna())):
        price_and_returns = df[['Date', 'Daily price']]
        price_and_returns = price_and_returns[price_and_returns["Date"] > "2017-11-30"]
        price_and_returns = price_and_returns[price_and_returns["Date"] < "2018-12-01"]
        price_and_returns = pd.merge(SSE, price_and_returns, on='Date', how='left')
        price_and_returns= price_and_returns[['Date', 'Daily price']]
        price_and_returns['Returns'] = price_and_returns['Daily price'].pct_change(fill_method=None)
        price_and_returns= price_and_returns[np.isfinite(price_and_returns['Daily price'])]
    else:
        price_and_returns= None

    if (len(df['Volume of Stock'])>sum(df['Volume of Stock'].isna())):
        stockVol = df[['Unnamed: 3', "Volume of Stock"]]
        stockVol.columns = ['Date', "Volume of Stock"]
        stockVol = stockVol[stockVol["Date"] > "2017-11-30"]
        stockVol = stockVol[stockVol["Date"] < "2018-12-01"]
    else:
        stockVol = None

    if ((len(df['Bid'])>sum(df['Bid'].isna())) or (len(df['Ask'])>sum(df['Ask'].isna()))) :
        Bid = df[['Unnamed: 5',"Bid"]]
        Bid.columns = ['Date', "Bid"]
        Ask = df[['Unnamed: 7', "Ask"]]
        Ask.columns = ['Date', "Ask"]
        if (len(Bid)>=len(Ask)):
            bask = pd.DataFrame(pd.merge(Bid, Ask, how='left', on='Date'))
        else:
            bask = pd.DataFrame(pd.merge(Ask, Bid, how='left', on='Date'))
        bask = bask[bask["Date"] > "2017-11-30"]
        bask = bask[bask["Date"] < "2018-12-01"]
    else:
        bask = None

    if (len(df['Market Cap'])>sum(df['Market Cap'].isna())):
        markCap = df[['Unnamed: 9', "Market Cap"]]
        markCap.columns = ['Date', "Market Cap"]
        markCap = markCap[markCap["Date"] > "2017-11-30"]
        markCap = markCap[markCap["Date"] < "2018-12-01"]
    else:
        markCap = None

    return price_and_returns, stockVol, bask, markCap

## Save to pkl
def save_pkl():
    f = open("db/changed_price_and_returns.pkl", "wb")
    pkl.dump(priceReturns, f)
    f.close()

    f = open("db/changed_stock_volume.pkl", "wb")
    pkl.dump(stockVolume, f)
    f.close()

    f = open("db/changed_bid_and_ask.pkl", "wb")
    pkl.dump(bidAsk, f)
    f.close()

    f = open("db/changed_market_cap.pkl", "wb")
    pkl.dump(marketCap, f)
    f.close()

    f = open("db/changed_compnames.pkl", "wb")
    pkl.dump(compnames, f)
    f.close()

## Initialize memory
compnames = []
priceReturns = {}
stockVolume = {}
bidAsk = {}
marketCap = {}

#### load all data

## Process Anthony's data
files = ['A', 'B', 'C', 'D','EF', 'G', 'I', 'H']
for letter in files:
    file = 'db/Changed ' + letter + '.xlsx'
    names = xlrd.open_workbook(file).sheet_names()  ## get company names
    compnames.append(names)
    for name in names:
        print(name)
        priceReturns[name], stockVolume[name], bidAsk[name], marketCap[name] =read_in(name, 1)

## Process Nat's data
file = 'db/Nat_Data.xlsx'
names = xlrd.open_workbook(file).sheet_names()  ## get company names
compnames.append(names)
for name in names:
    print(name)
    priceReturns[name], stockVolume[name], bidAsk[name], marketCap[name] = read_in(name, 2)
compnames = [name for names in compnames for name in names]
save_pkl()
'''

def load_pkl(prefix):
    with open("db/" + prefix + "price_and_returns.pkl", 'rb') as f:
        a = pkl.load(f)
    f.close()

    with open("db/" + prefix + "stock_volume.pkl", 'rb') as f:
        b = pkl.load(f)
    f.close()

    with open("db/" + prefix + "bid_and_ask.pkl", 'rb') as f:
        c = pkl.load(f)
    f.close()

    with open("db/" + prefix + "market_cap.pkl", 'rb') as f:
        d = pkl.load(f)
    f.close()

    with open("db/" + prefix + "compnames.pkl", 'rb') as f:
        e = pkl.load(f)
    f.close()
    return a, b, c, d, e

priceReturns, stockVolume, bidAsk, marketCap, keys = load_pkl('changed_')
name = 'CHINA YANGTZE A (HK-C)'

def tradeWar():
    TW = list(pd.read_csv('db/Trade war.csv')['Name'])
    out = {}
    for key in keys:
        if (key in TW):
            out[key] = 1
        else:
            out[key] = 0
    return out
TW = tradeWar()

def checkNone(key):
    if (priceReturns[key]is None):
        return True
    elif (stockVolume[key] is None):
        return True
    elif (bidAsk[key] is None):
        return True
    elif (marketCap[key] is None):
        return True
    else:
        return False

def market_volume(inclusion, m, sse):
    ## Load data, and preprocess
    df = pd.DataFrame(pd.merge(sse, m, how='left', on='Date'))
    df = df [["Date", "Volume of Stock", "Volume of Index"]].fillna(0)
    df['months'] = [dates.month for dates in df["Date"]]

    ## group by months
    Vit = []
    Vmt =  []
    index = []
    year = [2017]*11
    year.append(2018)

    ## Monthy sum
    for i in range(1, 13, 1):
        index = df['months']==i
        Vit.append(sum(df['Volume of Stock'][index]))
        Vmt.append(sum(df['Volume of Index'][index]))

    ## readjust to Dec 2017 - Nov 2018
    month = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    Vit.insert(0, Vit.pop())
    Vmt.insert(0, Vmt.pop())

    out = pd.DataFrame({"Month":month,'Vit': Vit, 'Vmt': Vmt})
    Vi = np.mean(Vit[:6])
    Vm = np.mean(Vmt[:6])
    out["VRit"] = ((out['Vit']/out['Vmt']) * (Vm/Vi))
    return (out)

def market_liquidity (m):
    df = pd.merge(m, SSEBA, on='Date', how='left').dropna()
    index = df["Date"] <= inclusion
    ba = df["Bid"] - df["Ask"]
    df['month'] = [dates.month for dates in df['Date']]

    Sit = []
    Smt = []
    for i in range (1,13,1):
        index = df['month'] == i
        Sit.append(np.mean(ba[index]))
        Smt.append(np.mean(df['bidask'][index]))

    month = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    Sit.insert(0, Sit.pop())
    Smt.insert(0, Smt.pop())

    out = pd.DataFrame({"Month": month, 'Sit':Sit, 'Smt':Smt}).dropna()
    Sm = np.mean(out['Smt'][:6])
    Si = np.mean(out['Sit'][:6])
    out['Liq'] = (out['Sit']/out['Smt'])*(Sm/Si)
    return out

def standard_deviation (m):  ## Volatility per company
    before = m["Date"] <= inclusion
    returns = m['Returns']
    rave = np.std(returns[before])

    ri = pd.DataFrame({"Date": m["Date"], "r": returns})
    ri["month"] = [dates.month for dates in ri["Date"]]
    df = []
    for i in range(1,13,1):
        index = ri["month"] == i
        stdv = np.std(returns[index])
        diff = stdv - rave
        temp = [i, stdv, rave, diff]
        df.append(temp)

    df.insert(0, df.pop())
    df = pd.DataFrame(df,columns=["month", "std dev", "ave", "diff"])
    return df

def aggregate_regression():
    ls = []
    for key in keys:
        if not(checkNone(key)):
            temp = read_SSE()
            temp = pd.merge(temp, priceReturns[key], how='left', on='Date')
            temp = pd.merge(temp, stockVolume[key], how='left', on='Date')
            temp = pd.merge(temp, bidAsk[key], how='left', on='Date')
            temp = pd.merge(temp, marketCap[key], how='left', on='Date')
            temp['tradeWar'] = TW[key]
            ls.append(temp)
    df = pd.concat(ls)
    df = df[['Date', 'Returns', 'Volume of Stock', 'Market Cap', 'Index returns', 'tradeWar']].dropna()
    ri = df['Returns']
    vol = df['Volume of Stock']
    before = df["Date"] <= inclusion
    after = np.logical_not(before)
    D0 = before * 1
    D1 = after * 1
    cap = df['Market Cap']
    MP = df['Index returns']
    trdwar = df['tradeWar']

    x1 = D1  # Dummy variable
    x2 = np.log(vol)  # log of volume of stock traded
    x3 = np.multiply(np.log(vol), D1)  # stock of volume traded after inclusion
    x4 = np.log(cap)  # market cap of company
    x5 = MP  # Market performance
    x6 = trdwar  # Trade war dummy

    x = pd.DataFrame({"dummy": x1, "Vol": x2, "postVol": x3, "lncap": x4, "lnMP": x5, "TW": x6})
    x = sm.add_constant(x)
    y = ri

    lm = sm.OLS(y, x).fit()
    summary = lm.summary()
    with open("results/aggregate_regression.txt", "w") as f:
        f.write(str(summary))

    return summary ## summary of regression
agg_summary = aggregate_regression()

def return_regression(PR, SV, MC):
    df = pd.DataFrame(pd.merge(SSE, PR, on='Date', how='left'))
    df = pd.DataFrame(pd.merge(df, SV, on='Date', how='left'))
    df = pd.DataFrame(pd.merge(df, MC, on='Date', how='left'))
    df = df[['Date', 'Returns', 'Volume of Stock', 'Market Cap', 'Index returns']].dropna()
    ri = df['Returns']
    vol = df['Volume of Stock']
    before = pd.to_datetime(df["Date"], format='%Y-%m-%d')<= pd.to_datetime(inclusion, format='%d-%m-%y')
    after = np.logical_not(before)
    D0 = before*1
    D1 = after*1
    cap = df['Market Cap']
    MP = df['Index returns']


    x1 = D1 # Dummy variable
    x2 = np.log(vol) # log of volume of stock traded
    x3 = np.multiply(np.log(vol), D1) # stock of volume traded after inclusion
    x4 = np.log(cap) # market cap of company
    x5 = MP # Market performance
    x6 = TW[key] # Trade war dummy

    x = pd.DataFrame({"dummy": x1, "Vol": x2, "postVol": x3, "lncap": x4, "MP": x5})
    y = ri

    lm = linear_model.LinearRegression()
    lm.fit(x,y)
    # predictions = lm.predict(x) ## for reference

    coefs = list(x)
    coefs.append('intercept')
    coefs.append('score')
    out = list(lm.coef_) # Vol, dmmy, lnMP, lncap, postVol
    out.append(lm.intercept_)
    out.append(lm.score(x,y))
    out = dict(zip(coefs,out))

    return out ## coefs, intercept, score

## prep memory
marketVol = {}
MLiq = {}
staDev = {}
regs = {}
none = []

for key in keys:
    if (not(checkNone(key))):
        #marketVol[key] = market_volume(inclusion, stockVolume[key], SSE)
        MLiq[key] = market_liquidity(bidAsk[key])
        #staDev[key] = standard_deviation(priceReturns[key])
        #regs[key] = return_regression(priceReturns[key], stockVolume[key], marketCap[key])
    else:
        none.append(key)

## Aggregate MV
def aggMV():
    agg = {}
    for k, v in marketVol.items():
        if (sum(np.isfinite(v['VRit'][6:])) > 1):
            agg[k] = np.mean(v['VRit'][6:])
    alpha, pval = stats.ttest_1samp(list(agg.values()), 1)
    names, aveVRIT= [], []
    for k, v in agg.items():
        names.append(k)
        aveVRIT.append(v)
    df = pd.DataFrame({'names':names, 'VRITave':aveVRIT, 'alpha': alpha, 'pval':pval})
    return df, alpha, pval

aggMarketVol, alpha, pval = aggMV()
H1a = [alpha, pval]
aggMarketVol.sort_values(by='VRITave', inplace=True, ascending=False)
aggMarketVol.to_csv('results/changed_H1a_aggregated_market_volume.csv', index=False)
#aggVRIT = (np.mean(list(aggMarketVol.values()))-1)/np.std(list(aggMarketVol.values()))

agg_over_one, agg_under_one = {}, {}
for i in range(len(aggMarketVol)):
    if ((aggMarketVol['VRITave'][i]>1) and (np.isfinite([aggMarketVol['VRITave'][i]]))):
        agg_over_one[aggMarketVol['names'][i]] = aggMarketVol['VRITave'][i]
    elif ((aggMarketVol['VRITave'][i]<1) and (np.isfinite([aggMarketVol['VRITave'][i]]))):
        agg_under_one[aggMarketVol['names'][i]] = aggMarketVol['VRITave'][i]

def tradingVol(MV):
#    VRit = np.mean(MV['VRit'][6:])
#    sd = np.std(MV['VRit'][6:])
#    alpha = ((VRit - 1) / sd)
    t, pval = stats.ttest_1samp(MV['VRit'][6:], 1)
    return [t, pval]

VRIT, VRIT_over_one, VRIT_nega_one  = {}, {}, {} ## Alphas and Pvals
for k, v in marketVol.items():
    VRIT[k] = tradingVol(v)
    if (VRIT[k][0] > 1):
        VRIT_over_one[k] = VRIT[k]
    elif(VRIT[k][0] < -1):
        VRIT_nega_one[k] = VRIT[k]

def aggregate_VRIT():
    name, alpha, pval, vrit6, vrit7, vrit8, vrit9, vrit10, vrit11, VRITave= [], [], [], [], [], [], [], [], [], []
    for k, v in VRIT.items():
        name.append(k)
        alpha.append(v[0])
        pval.append(v[1])
    for n in name:
        vrit6.append(marketVol[n]['VRit'][6])
        vrit7.append(marketVol[n]['VRit'][7])
        vrit8.append(marketVol[n]['VRit'][8])
        vrit9.append(marketVol[n]['VRit'][9])
        vrit10.append(marketVol[n]['VRit'][10])
        vrit11.append(marketVol[n]['VRit'][11])
        VRITave.append(np.mean(marketVol[n]['VRit'][6:]))
    df = pd.DataFrame({'name':name, 'alpha':alpha, 'pval':pval, 'vrit6':vrit6, 'vrit7':vrit7, 'vrit8':vrit8, 'vrit9':vrit9, 'vrit10':vrit10, 'vrit11':vrit11, 'VRITave':VRITave})
    return df

H1b = aggregate_VRIT()
H1b.sort_values(by='pval', inplace=True)
H1b.to_csv('results/changed_H1b_marketVol.csv', index=False)

def marketLiq(ML):
    t, pval = stats.ttest_1samp(ML['Liq'][6:], 1)
    LiqAve = np.mean(ML['Liq'][6:])
    return [t, pval, LiqAve]

names, alphas, pvals, LiqAves = [], [], [], []
for k, v in MLiq.items():
    names.append(k)
    alpha, pval, LiqAve= marketLiq(v)
    alphas.append(alpha)
    pvals.append(pval)
    LiqAves.append(LiqAve)

H1c = pd.DataFrame({'name':names, 'alpha':alphas, 'pval':pvals, 'LiqAve':LiqAves}).dropna()
H1c.sort_values(by='pval', inplace=True)

rem1 = 'HANGZHOU ROBAM A (HK-C)'
rem2 = 'IFLYTEK CO A (HK-C)'
H1c = H1c[H1c['name'] != rem1]
H1c = H1c[H1c['name'] != rem2]

alpha, pval = stats.ttest_1samp(H1c['LiqAve'], 1)
H1c['tot_alpha'] = alpha
H1c['tot_pval'] = pval
H1c.to_csv('results/changed_H1c_marketVol.csv')

def agg_SD():
    ave_diff, name = [], []
    for k, v in staDev.items():
        if (np.isfinite([np.mean(v['diff'][6:])])):
            ave_diff.append(np.mean(v['diff'][6:]))
            name.append(k)
    df = pd.DataFrame({'name': name, 'aveDiff':ave_diff})
    return df

agg_stadev = agg_SD()
alpha, pval = stats.ttest_1samp(agg_stadev['aveDiff'], 0)
agg_stadev['alpha'] = alpha
agg_stadev['pval'] = pval
agg_stadev.sort_values(by='aveDiff', inplace=True, ascending=False)
agg_stadev.to_csv('results/changed_H2a_intercompany_sd.csv', index=False)


def risks(SD):
    sdt = SD['diff'][6:]
#    sdt = np.mean(sdt)
#    std_sdt = np.std(sdt)
#    alpha = (sdt-1)/std_sdt
    t, pval = stats.ttest_1samp(sdt, 0)
    return [t, pval]

volatility, volatility_over_one, volatility_nega_one = {}, {}, {}

for k, v in staDev.items():
    volatility[k] = risks(v)
    if (volatility[k][0]>1):
        volatility_over_one[k] = volatility[k]
    elif (volatility[k][0]< -1):
        volatility_nega_one[k] = volatility[k]

## Investigate nans
def aggregate_volatility():
    nans, name, alpha, pval = [], [], [], []
    diffAve, diff6, diff7, diff8, diff9, diff10, diff11 = [], [], [], [], [], [], []
    for k, v in volatility.items():
        if (np.isnan(v[0])):
            nans.append(k)
        else:
            name.append(k)
            alpha.append(v[0])
            pval.append(v[1])
            diff6.append(staDev[k]['diff'][6])
            diff7.append(staDev[k]['diff'][7])
            diff8.append(staDev[k]['diff'][8])
            diff9.append(staDev[k]['diff'][9])
            diff10.append(staDev[k]['diff'][10])
            diff11.append(staDev[k]['diff'][11])
            diffAve.append(np.mean(staDev[k]['diff'][6:]))
    df = pd.DataFrame({'name':name, 'alpha':alpha, 'pvalue':pval, 'diff6':diff6, 'diff7':diff7, 'diff8':diff8, 'diff9':diff9, 'diff10':diff10, 'diff11':diff11, 'diffAve':diffAve})
    return nans, df

nans, H2b = aggregate_volatility()
H2b.sort_values(by='pvalue', inplace=True)
H2b.to_csv('results/changed_H2b_intracompany_sd.csv', index=False)

def aggregate_regs():
    comp, x1, x2, x3, x4, x5, x6, intercept, score = [], [], [], [], [], [], [], [], []
    for k, v in regs.items():
        comp.append(k)
        x1.append(v['dummy'])
        x2.append(v['Vol'])
        x3.append(v['postVol'])
        x4.append(v['lncap'])
        x5.append(v['lnMP'])
        intercept.append(v['intercept'])
        score.append(v['score'])
    df = pd.DataFrame({'name':comp, 'b1': x1, 'b2':x2, 'b3':x3, 'b4':x4, 'b5':x5, 'intercept':intercept, 'score':score})
    alpha, pval = stats.ttest_1samp(df['b1'], 0)
    df['alpha'] = alpha
    df['pval'] = pval
    return df

agg_regs = aggregate_regs()
ave_b1 = np.mean(agg_regs['b1'])
agg_regs.to_csv('results/changed_H3_aggregated_regression_results.csv', index=False)
print ('END')