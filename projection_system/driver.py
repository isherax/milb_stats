import build_data
import milb_model
import numpy as np
import os
import pandas as pd


def merge_projections(year_range):
    positions = ['C', '1B', '2B', 'SS', '3B', 'LF', 'CF', 'RF']
    
    for year in year_range:
        bat_df = pd.read_csv('projections/{0}_bat_projections.csv'.format(year))
        pit_df = pd.read_csv('projections/{0}_pit_projections.csv'.format(year))
        bat_df['position'] = ''
        pit_df['position'] = 'P'
        
        for pos in positions:
            bat_df['position'] = np.where((bat_df[pos] > 0) & (bat_df['position'] != ''), bat_df['position'] + '/' + pos, bat_df['position'])
            bat_df['position'] = np.where((bat_df[pos] > 0) & (bat_df['position'] == ''), pos, bat_df['position'])
            
        pit_df['position'] = np.where(pit_df['O/G'] > 7.5, 'SP', pit_df['position'])
        pit_df['position'] = np.where((pit_df['O/G'] <= 7.5) & (pit_df['O/G'] > 4.5), 'MIRP', pit_df['position'])
        pit_df['position'] = np.where(pit_df['O/G'] <= 4.5, 'SIRP', pit_df['position'])
                    
        bat_df = bat_df[['Name', 'playerid', 'position', 'mlb_team', 'w0', 'w1', 'w2', 'w3', 'w4+', 'wxWAR']]
        pit_df = pit_df[['Name', 'playerid', 'position', 'mlb_team', 'w0', 'w1', 'w2', 'w3', 'w4+', 'wxWAR']]
        
        out_df = pd.concat([bat_df, pit_df])
        out_df.sort_values(by='wxWAR', ascending=False, inplace=True)
        out_df.to_csv('projections/{0}_cmb_projections.csv'.format(year), index=False)
        

def clear_exp_folder(year_range):
    exp_folder = 'reports/exp_data'
    
    for file in os.listdir(exp_folder):
        year = int(file.split('_')[0])
        
        if year >= min(year_range) and year <= max(year_range):
            os.remove(os.path.join(exp_folder, file))
        
        
def get_new_reports(year_range):
    for year in year_range:
        build_data.get_reports(year, stats='bat', qual='0', stat_type='all', ind='1')
        build_data.get_reports(year, stats='pit', qual='0', stat_type='all', ind='1')
        build_data.get_pos_data(year)
        build_data.get_elig(year, stats='bat', qual='150')
        build_data.get_elig(year, stats='pit', qual='50')
    
    
def run_projections(stat_type, features, year_range):
    for i, year in enumerate(year_range):
        model = milb_model.milb_model(stat_type, features)

        if year == 2020:
            df = pd.read_csv('projections/2019_{0}_projections.csv'.format(stat_type))
            df.to_csv('projections/2020_{0}_projections.csv'.format(stat_type), index=False)
            bat_exp = pd.read_csv('reports/exp_data/2019_{0}.csv'.format(stat_type))
            bat_exp.to_csv('reports/exp_data/2020_{0}.csv'.format(stat_type), index=False)
        elif year == 2006:
            model.train_models(verbose=False)
            model.run_models(year, False)
        else:
            model.train_models()
            model.run_models(year)


if __name__ == '__main__':
    batting_features = ['Age', 'Exp', 'BB%', 'K%', 'Spd', 'ISO', 'wRC+']
    pitching_features = ['Age', 'Exp', 'O/G', 'K-BB%', 'WHIP', 'HR/9']
    projection_range = range(2014, 2023)    
    
    get_new_reports(projection_range)
    clear_exp_folder(projection_range)
    run_projections('bat', batting_features, projection_range) 
    run_projections('pit', pitching_features, projection_range)
    merge_projections(projection_range[1:])
        