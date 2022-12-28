import openturns as ot
import pandas as pd
import numpy as np

def best_dist_docs_continuous(sample):
    sample = sample
    sample = sample.to_numpy()
    sample = sample[~np.isnan(sample)]
    sample = sample.reshape((-1, 1))

    tested_factories = ot.DistributionFactory.GetContinuousUniVariateFactories()
    best_model, best_bic = ot.FittingTest.BestModelBIC(sample, tested_factories)

    return str(best_model)


def best_dist_docs_discrete(sample):
    sample = sample
    sample = sample.to_numpy()
    sample = sample[~np.isnan(sample)]
    sample = sample.reshape((-1, 1))

    tested_factories = ot.DistributionFactory.GetDiscreteUniVariateFactories()
    best_model, best_bic = ot.FittingTest.BestModelBIC(sample, tested_factories)

    return str(best_model)



numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# Example section:
act = pd.read_csv("C:\\Users\\Jens Harbers\\Documents\\ACT_Collar00869_20200316171823.csv", sep=";",decimal=",", encoding="latin-1")
act["timestamp"] = pd.to_datetime(act['UTC_Date'] +" "+act['UTC_Time'], format="%d.%m.%Y %H:%M:%S")

df2 = act.select_dtypes(include=numerics)
df2.reset_index(drop=True)
# choose columns with at least two unique values
ids = df2.nunique().values>1
df2 = df2.loc[:,ids]

dist_list = []

for i in np.arange(len(df2.columns)):
    dist_list.append(best_dist_docs_continuous(df2.iloc[:,i]))

desc = df2.describe().T
desc["distributions"] = dist_list
desc["dtype"] = df2.dtypes
desc["kurtosis"]= df2.kurtosis()
desc["skew"]= df2.skew()
if len(act.select_dtypes(exclude=numerics)) > 0:
    df3 = act.select_dtypes(exclude=numerics)
    df3 = df3.apply(pd.Categorical)
    desc2 = df3.describe().T
    desc2["distributions"] = np.repeat(np.nan,len(act.select_dtypes(exclude=numerics).columns))
    desc2["dtype"] = df3.dtypes
    descs = pd.concat([desc,desc2])
