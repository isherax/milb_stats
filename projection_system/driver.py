import milb_model
import pandas as pd


def merge_projections(year):
    bat_df = pd.read_csv('projections/{0}_bat_projections.csv'.format(year))
    pit_df = pd.read_csv('projections/{0}_pit_projections.csv'.format(year))
    bat_df = bat_df[['Name', 'playerid', 'w-1',	'w0', 'w1', 'w2', 'w3', 'w4+', 'wxWAR']]
    pit_df = pit_df[['Name', 'playerid', 'w-1',	'w0', 'w1', 'w2', 'w3', 'w4+', 'wxWAR']]
    
    out_df = pd.concat([bat_df, pit_df])
    out_df.sort_values(by='wxWAR', ascending=False, inplace=True)
    out_df.to_csv('projections/{0}_cmb_projections.csv'.format(year), index=False)
    
    
def run_projections(stat_type, features, year_range):
    model = milb_model.milb_model(stat_type, features)
    
    for i, year in enumerate(year_range):
        model.train_models()
        
        if year == 2020:
            df = pd.read_csv('projections/2019_{0}_projections.csv'.format(stat_type))
            df.to_csv('projections/2020_{0}_projections.csv'.format(stat_type), index=False)
        else:
            if i == 0:
                model.run_models(year, False)
            else:
                model.run_models(year)


if __name__ == '__main__':
    projection_range = range(2015, 2022)
    merge_range = range(2016, 2022)
    batting_features = ['Age', 'BB%', 'K%', 'Spd', 'ISO', 'wRC+']
    pitching_features = ['Age', 'O/G', 'K-BB%', 'WHIP']
    
    # run_projections('bat', batting_features, projection_range)
    # run_projections('pit', pitching_features, projection_range)
    
    for year in merge_range:
        merge_projections(year)
    