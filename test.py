import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

df = pd.DataFrame([[np.inf, 2, np.inf, 0],
                    [3, 4, -np.inf, 1],
                    [7, np.inf, -np.inf, 5],
                    [np.inf, 3, np.inf, 4]],
                    columns=list('ABCD'))


print(df)


df = df.replace([np.inf, -np.inf], np.nan)
print(df)

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_mean.fit(df)
print(imp_mean.transform(df))



