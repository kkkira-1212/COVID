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

    exclude_cols = ['Latitude', 'Longitude', 'lat', 'lon', 'Lat', 'Lon']
    for col in exclude_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.sort_values(['State', 'Date']).reset_index(drop=True)

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df

def engineer_features(df):
    df = df.copy()
    df['NewDeaths_return'] = df.groupby('State')['NewDeaths'].transform(lambda x: x.pct_change().fillna(0))
    df['NewCases_MA7'] = df.groupby('State')['NewCases'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['NewDeaths_MA7'] = df.groupby('State')['NewDeaths'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    
    def safe_growth_rate(x):
        shifted = x.shift(7)
        result = (x - shifted) / shifted
        result = result.replace([np.inf, -np.inf], 0)
        return result.fillna(0)
    
    df['Cases_GrowthRate'] = df.groupby('State')['NewCases'].transform(safe_growth_rate)
    
    df['NewDeaths_return'] = df['NewDeaths_return'].replace([np.inf, -np.inf], 0)
    df['Cases_GrowthRate'] = df['Cases_GrowthRate'].replace([np.inf, -np.inf], 0)
    
    return df

def group_by_state(df, freq='D', agg_map=None):
    df = df.copy()
    
    exclude_cols = ['Latitude', 'Longitude', 'lat', 'lon', 'Lat', 'Lon']
    for col in exclude_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
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
            
            week_agg_map = agg_map.copy()
            if 'outbreak_label' in d.columns:
                week_agg_map['outbreak_label'] = 'max'
            
            d = d.resample('W-SUN').agg(week_agg_map).reset_index()
            d = d.rename(columns={'Date': 'WeekStart'})
            
            d['NewDeaths_return'] = d['NewDeaths'].pct_change().fillna(0)
            d['NewCases_MA7'] = d['NewCases'].rolling(7, min_periods=1).mean()
            d['NewDeaths_MA7'] = d['NewDeaths'].rolling(7, min_periods=1).mean()
            
            shifted = d['NewCases'].shift(7)
            d['Cases_GrowthRate'] = ((d['NewCases'] - shifted) / shifted).replace([np.inf, -np.inf], 0).fillna(0)
            
            d['NewDeaths_return'] = d['NewDeaths_return'].replace([np.inf, -np.inf], 0)
            d['Cases_GrowthRate'] = d['Cases_GrowthRate'].replace([np.inf, -np.inf], 0)
            
            if 'outbreak_label' in d.columns:
                d['outbreak_label'] = d['outbreak_label'].astype(int)
            
            data[s] = d
        return data
    raise ValueError