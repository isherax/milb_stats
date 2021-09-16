import fangraphs_scraper as fs
import pandas as pd
from io import StringIO


def get_reports(year, stats, qual, stat_type, ind):
    leaderboard = fs.get_milb_leaderboard(season=year, stats=stats, lg='2,4,5,6,7,8,9,10,11,14', 
                                          qual=qual, stat_type=stat_type, ind=ind)
    df = pd.read_csv(StringIO(str(leaderboard, 'utf-8')))
    df['Level'] = df['Team'].str.extract('.*\((.*)\).*')
    
    rminus_leaderboard = fs.get_milb_leaderboard(season=year, stats=stats, lg='16,17', 
                                                 qual=qual, stat_type=stat_type, ind=ind)
    rminus_df = pd.read_csv(StringIO(str(rminus_leaderboard, 'utf-8')))
    rminus_df['Level'] = 'R-'
    
    dsl_leaderboard = fs.get_milb_leaderboard(season=year, stats=stats, lg='30', 
                                              qual=qual, stat_type=stat_type, ind=ind)
    dsl_df = pd.read_csv(StringIO(str(dsl_leaderboard, 'utf-8')))
    dsl_df['Level'] = 'DSL'
    
    df = df.append(rminus_df).append(dsl_df)
    percents = [i for i in df.columns if '%' in i]
    for percent in percents:
        df[percent] = df[percent].str.rstrip('%').astype('float') / 100.0
    
    df = pd.get_dummies(df, columns=['Level'])
    df.to_csv('reports/{0}_milb_{1}.csv'.format(year, stats), index=False)
    

def get_mlb_data(start_year, stats, stat_type):
    leaderboard = fs.get_mlb_leaderboard(season='2021', stats=stats, qual='0', stat_type=stat_type, season1=start_year)
    df = pd.read_csv(StringIO(str(leaderboard, 'utf-8')))
    df.to_csv('mlb_{0}.csv'.format(stats), index=False)


def get_historical_reports(years):
    for year in years:
        get_reports(year=year, stats='bat', qual='y', stat_type='c,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50', ind='1')
        # get_reports(season=year, stats='pit', qual='y', stat_type='c,6,7,8,13,14,17,28,29,42,48', ind='1')


if __name__ == '__main__':
    years = range(2007, 2022)
    get_historical_reports(years)
    # get_mlb_data(years[0], stats='bat', stat_type='c,58')
        