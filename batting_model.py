import pandas as pd
import statsmodels.api as sm
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


def train_model(years, training_features):
    milb_sample = pd.DataFrame()
    
    for year in sample_years:
        year_sample = pd.read_csv('reports/{0}_milb_bat.csv'.format(year))
        year_sample['Year'] = year
        milb_sample = milb_sample.append(year_sample)
    
    mlb_sample = pd.read_csv('mlb_bat.csv', dtype={'playerid': str})
    mlb_sample = mlb_sample[['WAR', 'playerid']]
    
    milb_sample = milb_sample[milb_sample['Age'] < 30]
    milb_sample['NewVar'] = milb_sample['Strikes'] / milb_sample['Pitches'] - milb_sample['SwStr%']
    milb_sample = pd.merge(milb_sample, mlb_sample, on='playerid')
    milb_sample.fillna(value=0, inplace=True)

    x = milb_sample[training_features]
    scaler = StandardScaler()
    # x_scaled = scaler.fit_transform(x)
    poly = PolynomialFeatures(2)
    # x_t = poly.fit_transform(x_scaled)
    # new_cols = poly.get_feature_names(x.columns)
    # x_tdf = pd.DataFrame(x_t, columns=new_cols)
    # x_tdf = sm.add_constant(x_tdf)
    y = milb_sample['WAR']
    model = sm.OLS(y, x).fit()
    print(model.summary())
        
    return model, scaler, poly


def run_projections(year, model, scaler, poly, training_features):
    projection_input = pd.read_csv('reports/{0}_milb_bat.csv'.format(year))
    
    x = projection_input[training_features]
    # x_scaled = scaler.transform(x)
    # x_t = poly.transform(x_scaled)
    # new_cols = poly.get_feature_names(x.columns)
    # x_tdf = pd.DataFrame(x_t, columns=new_cols)
    # x_tdf = sm.add_constant(x_tdf)
    projection_output = projection_input.copy()
    projection_output['WAR'] = model.predict(x)
                
    projection_output.to_csv('projections/{0}_bat_projections.csv'.format(year), index=False)


if __name__ == '__main__':
    sample_years = range(2007, 2015)
    training_features = ['Age', 'BB%', 'K%', 'Spd', 'wRC+', 
                         'Level_AAA', 'Level_AA', 'Level_A+', 'Level_A', 'Level_R-', 'Level_DSL']
    
    model, scaler, poly = train_model(sample_years, training_features)
    run_projections(2021, model, scaler, poly, training_features)
