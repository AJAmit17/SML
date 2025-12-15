import pandas as pd
import numpy as np
import seaborn as sns

df = sns.load_dataset('titanic')

# WOE & IV function
def woe_iv(data, feature, target):
    temp = data[[feature, target]].dropna()
    groups = temp.groupby(feature)[target].agg(['count', 'sum'])
    
    good = groups['count'] - groups['sum']
    bad = groups['sum']
    
    good_pct = good / good.sum()
    bad_pct = bad / bad.sum()
    
    woe = np.log(good_pct / bad_pct)
    iv = ((good_pct - bad_pct) * woe).sum()
    
    return woe.to_dict(), iv

# Apply WOE to features and calculate IV
for col in ['sex', 'pclass']:
    woe_map, iv = woe_iv(df, col, 'survived')
    df[col + '_woe'] = df[col].map(woe_map)
    print(f"{col} - IV: {iv:.4f}")

print("\n", df[['sex', 'sex_woe', 'pclass', 'pclass_woe']].head())
