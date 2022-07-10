import fangraphs_scraper as fs
import pandas as pd
from io import StringIO


def get_reports(year, stats, qual, stat_type, ind):
    if stats == 'bat' and stat_type == 'all':
        stat_type = 'c,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50'
    elif stats == 'pit' and stat_type == 'all':
        stat_type = 'c,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52'
    
    leaderboard = fs.get_milb_leaderboard(season=year, stats=stats, lg='2,4,5,6,7,8,9,10,11,13,14', qual=qual, stat_type=stat_type, ind=ind)
    df = pd.read_csv(StringIO(str(leaderboard, 'utf-8')))
    df['Level'] = df['Team'].str.extract('.*\((.*)\).*')
    
    rplus_leaderboard = fs.get_milb_leaderboard(season=year, stats=stats, lg='15,18', qual=qual, stat_type=stat_type, ind=ind)
    rplus_df = pd.read_csv(StringIO(str(rplus_leaderboard, 'utf-8')))
    rplus_df['Level'] = 'R+'
    
    rminus_leaderboard = fs.get_milb_leaderboard(season=year, stats=stats, lg='16,17', qual=qual, stat_type=stat_type, ind=ind)
    rminus_df = pd.read_csv(StringIO(str(rminus_leaderboard, 'utf-8')))
    rminus_df['Level'] = 'R-'
    
    dsl_leaderboard = fs.get_milb_leaderboard(season=year, stats=stats, lg='30', qual=qual, stat_type=stat_type, ind=ind)
    dsl_df = pd.read_csv(StringIO(str(dsl_leaderboard, 'utf-8')))
    dsl_df['Level'] = 'DSL'
    
    if len(rplus_leaderboard) > 0:
        df = df.append(rplus_df).append(rminus_df).append(dsl_df)
    else:
        df = df.append(rminus_df).append(dsl_df)
        
    percent_cols = [i for i in df.columns if df[i].astype(str).str.contains('%').any()]
    for col in percent_cols:
        df[col] = df[col].str.rstrip('%').astype('float') / 100.0
    
    df = pd.get_dummies(df, columns=['Level'])
    df.to_csv('reports/{0}_milb_{1}.csv'.format(year, stats), index=False)
    

def get_pos_data(year):
    positions = ['C', '1B', '2B', 'SS', '3B', 'LF', 'CF', 'RF']
    
    for position in positions:
        pos_lower = position.lower()
        leaderboard = fs.get_milb_leaderboard(season=year, pos=pos_lower, stats='bat', qual='0', stat_type='c,4', lg='2,4,5,6,7,8,9,10,11,13,14,16,17,30')
        df = pd.read_csv(StringIO(str(leaderboard, 'utf-8')))
        df.to_csv('reports/position_data/{0}_{1}.csv'.format(year, position), index=False)
    

def get_mlb_data(start_year, stats, stat_type):
    leaderboard = fs.get_mlb_leaderboard(season='2021', stats=stats, ind='1', qual='0', stat_type=stat_type, season1=start_year, age='14,29')
    df = pd.read_csv(StringIO(str(leaderboard, 'utf-8')))
    df.to_csv('mlb_{0}.csv'.format(stats), index=False)


def get_historical_reports(years, stats, stat_type):
    if stats == 'bat' and stat_type == 'all':
        stat_type = 'c,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50'
    elif stats == 'pit' and stat_type == 'all':
        stat_type = 'c,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52'
    for year in years:
        get_reports(year=year, stats=stats, qual='0', stat_type=stat_type, ind='1')


def agg_peak(x):
    list_x = list(x)
    list_x.sort(reverse=True)
    
    if len(list_x) >= 3:
        return sum(list_x[0:3]) / 3
    else:
        return sum(list_x) / 3


def group_mlb_data(stats):
    df = pd.read_csv('mlb_{0}.csv'.format(stats))
    grouped_df = df.groupby('playerid')['WAR'].agg(agg_peak)
    grouped_df.to_csv('peak_mlb_{0}.csv'.format(stats))
    

def get_elig(year, stats, qual):
    min_year = 2000
    leaderboard = fs.get_mlb_leaderboard(season=year, pos='all', stats=stats, qual=qual, stat_type='c,6', season1=min_year)
    df = pd.read_csv(StringIO(str(leaderboard, 'utf-8')))
    df.to_csv('eligibility/{0}_elig_{1}.csv'.format(year, stats), index=False)
    

if __name__ == '__main__':
    for year in range(2014, 2023):
        get_reports(year, stats='bat', qual='0', stat_type='all', ind='1')
        get_reports(year, stats='pit', qual='0', stat_type='all', ind='1')
        get_pos_data(year)
        get_elig(year, stats='bat', qual='150')
        get_elig(year, stats='pit', qual='50')
        