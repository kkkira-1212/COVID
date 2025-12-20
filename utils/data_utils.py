import pandas as pd
import numpy as np
agg_map = {
    'NewCases': 'sum',
    'NewDeaths': 'sum',
    'Patience_Count': 'sum',
    'Vax_AllDoses': 'sum',
    'Vax_Dose1': 'sum',
    'Vax_Dose2': 'sum',
    'Vax_Dose3': 'sum',
    'Vax_Dose4': 'sum',
    'Hosp_Count': 'sum',
    'Hosp_Deaths': 'sum',

    'Ct_Value': 'mean',
    'Stringency_Index': 'mean',
    'Aver_Hosp_Stay': 'mean',
    'TotalDeaths_by_TotalCases': 'mean',
    'Hosp_Death_Rate': 'mean',

    'TotalCases': 'last',
    'TotalDeaths': 'last',
    'TotalCases_100k_inhab': 'last',
    'TotalDeaths_100k_inhab': 'last',
}

def loader(file_path):
    df = pd.read_excel(file_path)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    if 'State' in df.columns:
        df = df.dropna(subset=['State'])

    df = df.sort_values(['State', 'Date']).reset_index(drop=True)

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df

def engineer_features(df):
    df = df.copy()
    df['NewDeaths_return'] = df.groupby('State')['NewDeaths'].transform(lambda x: x.pct_change().fillna(0))
    df['NewCases_MA7'] = df.groupby('State')['NewCases'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['NewDeaths_MA7'] = df.groupby('State')['NewDeaths'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['Cases_GrowthRate'] = df.groupby('State')['NewCases'].transform(lambda x: (x - x.shift(7)) / x.shift(7)).fillna(0)
    return df

def group_by_state(df, freq='D', agg_map=None):
    df = df.copy()
    states = sorted(df['State'].unique())
    data = {}
    if freq == 'D':
        for s in states:
            d = df[df['State'] == s].sort_values('Date').reset_index(drop=True)
            data[s] = d
        return data
    if freq == 'W':
        for s in states:
            d = df[df['State'] == s].set_index('Date')
            d = d.resample('W-SUN').agg(agg_map).reset_index()
            d = d.rename(columns={'Date': 'WeekStart'})
            data[s] = d
        return data
    raise ValueError