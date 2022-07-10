import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def test_models(years):
    total_all_corr = 0
    total_mlb_corr = 0
    
    for year in years:
        pitting_output = pd.read_csv('projections/{0}_pit_projections.csv'.format(year))
        mlb_pit_results = pd.read_csv('peak_mlb_pit.csv')
        mlb_pit_results['playerid'] = mlb_pit_results['playerid'].astype(str)
        
        pitting_output_all = pitting_output.merge(mlb_pit_results, on='playerid', how='left')
        pitting_output_all.fillna(value=0, inplace=True)
        pitting_output_war = pitting_output_all.sort_values(by='WAR', ascending=False)
        pitting_output_all = pitting_output_all.head(100)
        pitting_output_war = pitting_output_war.head(100)
        pitting_output_all.to_csv('test.csv')
        
        total_all_corr += pitting_output_all['wxWAR'].corr(pitting_output_all['WAR'])
        total_mlb_corr += pitting_output_war['wxWAR'].corr(pitting_output_war['WAR'])
    
    print('T100 correlation: ', total_all_corr / len(years))
    print('A100 correlation: ', total_mlb_corr / len(years))
        

class milb_model:
    def __init__(self, stat_type, features, years=None, levels=None, war_buckets=None, regressor_min=None, denom_min=None):
        self.stat_type = stat_type
        self.features = features.copy()
        self.models = {}
        self.bucket_means = {}
        self.milb_sample = pd.DataFrame()
        self.flat_levels = []
        self.static_vars = ['Age', 'Exp']
        self.positions = ['C', '1B', '2B', 'SS', '3B', 'LF', 'CF', 'RF']
        
        if stat_type == 'bat':
            self.denom = 'PA'
            self.features.extend(self.positions)
        elif stat_type == 'pit':
            self.denom = 'TBF'
        
        if years:
            self.years = years
        else:
            self.years = range(2006, 2014)
            
        if levels:
            self.level_groups = levels
        else:
            self.level_groups = {'high': ['Level_AAA', 'Level_AA'], 'mid': ['Level_A+', 'Level_A'], 'low': ['Level_A-', 'Level_R+', 'Level_R-', 'Level_DSL']}
            
        for key in self.level_groups.keys():
            self.flat_levels.extend(self.level_groups[key])
            
        if war_buckets:
            self.war_buckets = war_buckets
        else:
            self.war_buckets = {'0': [None, 0.5], '1': [0.5, 1.5], '2': [1.5, 2.5], '3': [2.5, 3.5], '4+': [3.5, None]}
            
        if regressor_min:
            self.regressor_min = regressor_min
        else:
            self.regressor_min = 20
            
        if denom_min:
            self.denom_min = denom_min
        else:
            self.denom_min = 400
            
            
    def create_buckets(self, df):
        for key in self.war_buckets.keys():
            bucket = self.war_buckets[key]
            
            if bucket:
                lower_bound = bucket[0]
                upper_bound = bucket[1]
                in_lower = False
                in_upper = False
                
                if lower_bound is None or df['WAR'] > lower_bound:
                    in_lower = True
                if upper_bound is None or df['WAR'] <= upper_bound:
                    in_upper = True
                    
                if in_lower and in_upper:
                    return key
                
        return '0'
    
    
    def calc_xwar(self, df):
        xwar = 0
        
        for key in self.war_buckets.keys():
            xwar += df[key] * self.bucket_means[key]
            
        return xwar
        
    
    def add_pos(self, df, year, pos):
        pos_df = pd.read_csv('reports/position_data/{0}_{1}.csv'.format(year, pos))
        p_series = df['playerid'].isin(pos_df['playerid'])
        new_df = df.copy()
        new_df[pos] = p_series.astype(int)
        
        return new_df
    
    def add_exp(self, df, year):
        default_path = 'reports/exp_data/{0}_{1}.csv'.format(year, self.stat_type)
        
        if os.path.exists(default_path):
            exp_df = pd.read_csv(default_path)
        else:
            exp_df = pd.read_csv('reports/exp_data/{0}_{1}.csv'.format(year - 1, self.stat_type))
            
        exp_df = exp_df.merge(df[['Name', 'playerid']], how='outer')
        exp_df.drop_duplicates(subset='playerid', inplace=True)
        exp_df['start_year'].fillna(value=year, inplace=True)
        exp_df.to_csv(default_path, index=False)
        
        df = df.merge(exp_df)
        df['Exp'] = year - df['start_year']
        
        if year > 2020:
            df['Exp'] = np.where(df['start_year'] > 2020, df['Exp'], df['Exp'] - 1)
        
        return df
        
    def train_models(self, verbose=False):
        for year in self.years:
            year_sample = pd.read_csv('reports/{0}_milb_{1}.csv'.format(year, self.stat_type))
            year_sample['Year'] = year
            
            if self.stat_type == 'bat':
                for position in self.positions:
                    year_sample = self.add_pos(year_sample, year, position)
                
                year_sample.loc[year_sample['C'] == 1, '1B'] = 0
                pos_sums = year_sample[self.positions].sum(axis=1)
                year_sample[self.positions] = year_sample[self.positions].div(pos_sums, axis=0)
                
            year_sample = self.add_exp(year_sample, year)
            self.milb_sample = self.milb_sample.append(year_sample)
        
        mlb_sample = pd.read_csv('peak_mlb_{0}.csv'.format(self.stat_type), dtype={'playerid': str})
        mlb_sample = mlb_sample[['WAR', 'playerid']]
        
        self.milb_sample = self.milb_sample[self.milb_sample[self.denom] >= 200]
        self.milb_sample = self.milb_sample[self.milb_sample['Age'] < 30]
        self.milb_sample = pd.merge(self.milb_sample, mlb_sample, on='playerid', how='left')
        self.milb_sample['WAR_bucket'] = self.milb_sample.apply(self.create_buckets, axis=1)
        self.milb_sample.to_csv('test.csv')
        self.milb_sample.fillna(value=0, inplace=True)
        
        if self.stat_type == 'pit':
            self.milb_sample['O'] = self.milb_sample['IP'].astype(int) * 3 + round(self.milb_sample['IP'] % 1, 1) * 10
            self.milb_sample['O/G'] = self.milb_sample['O'] / self.milb_sample['G']
            self.milb_sample = self.milb_sample[self.milb_sample['O/G'] >= 6]
        
        for key in self.war_buckets.keys():
            mean = self.milb_sample[self.milb_sample['WAR_bucket'] == key]['WAR'].mean()
            
            if mean > 0:
                self.bucket_means[key] = mean
            else:
                self.bucket_means[key] = 0
    
        for level_group in self.level_groups.keys():
            levels = self.level_groups[level_group]
            level_sample = self.milb_sample[self.milb_sample[levels].sum(axis=1) > 0]
            if level_group == 'high' and self.stat_type == 'bat':
                level_sample = level_sample[level_sample[self.denom] >= 400]
            elif level_group == 'mid' and self.stat_type == 'bat':
                level_sample = level_sample[level_sample[self.denom] >= 300]
            elif level_group != 'low' and self.stat_type == 'pit':
                level_sample = level_sample[level_sample[self.denom] >= 250]
                
            level_features = self.features.copy()
            level_features.extend(levels)
            x = level_sample[level_features]
            y = level_sample['WAR_bucket']
            
            model = LogisticRegression(solver='newton-cg', multi_class='multinomial').fit(x, y)
            if verbose:
                print(level_sample.groupby('WAR_bucket').describe())
                y_pred = model.predict(x)
                print('{0} model score: {1}'.format(level_group, round(balanced_accuracy_score(y, y_pred), 3)))
            self.models[level_group] = model
            
            
    def regressor_weights(self, df):
        denom_min = 10
        cols = list(self.war_buckets.keys())
        r_output = []
        
        for col in cols:
            r_output.append(((df[self.denom] * df[col]) + (denom_min * df['r{0}'.format(col)])) / (df[self.denom] + denom_min))
    
        return pd.Series(r_output)
    
    def weight_current(self, model_results, year):
        model_results = model_results[model_results[self.denom] > 0]
        new_cols = self.features
        
        for level in self.flat_levels:
            if level in model_results.columns:
                new_cols.append(level)
        
        weighted_mean = lambda x: np.average(x, weights=model_results.loc[x.index, self.denom])
        aggregate = dict.fromkeys(new_cols, weighted_mean)
        aggregate[self.denom] = np.sum
        
        for key in self.war_buckets.keys():
            aggregate[key] = weighted_mean
            aggregate['r{0}'.format(key)] = weighted_mean
        
        player_df = model_results[['Name', 'playerid', 'mlb_team']].copy()
        player_df.drop_duplicates(subset=['Name', 'playerid'], inplace=True)
        grouped_df = model_results.groupby(['Name', 'playerid']).agg(aggregate)
        grouped_df = player_df.merge(grouped_df, on='playerid')
        
        out_df = pd.DataFrame(grouped_df)
        out_df[list(self.war_buckets.keys())] = out_df.apply(self.regressor_weights, axis=1)
        out_df.drop(['r{0}'.format(x) for x in list(self.war_buckets.keys())], axis=1, inplace=True)
        out_df['xWAR'] = out_df.apply(self.calc_xwar, axis=1)
        out_df.reset_index(inplace=True)
        
        elig_df = pd.read_csv('eligibility/{0}_elig_{1}.csv'.format(year, self.stat_type))
        elig_df['playerid'] = elig_df['playerid'].astype(str)
        out_df = out_df[out_df['playerid'].isin(elig_df['playerid']) == False]
        out_df.sort_values(by='xWAR', ascending=False, inplace=True)
        
        return out_df
        
        
    def historical_weights(self, df):    
        if self.stat_type == 'bat':
            denom_min = 500
        else:
            denom_min = 250
        partial_denom = denom_min * 0.65
        cols = list(self.war_buckets.keys())
        cols.append('xWAR')
        w_output = []
        
        if df[self.denom] >= denom_min:
            for col in cols:
                w_output.append(df[col])
        elif df[self.denom] + df['t{0}'.format(self.denom)] >= denom_min:
            for col in cols:
                w_output.append((((df[self.denom] * df[col]) + (denom_min - df[self.denom]) * df['w{0}'.format(col)])) / denom_min)
        elif df[self.denom] + df['t{0}'.format(self.denom)] >= partial_denom:
            for col in cols:
                w_output.append((df[self.denom] * df[col] + (df['t{0}'.format(self.denom)] * df['w{0}'.format(col)])) / (df[self.denom] + df['t{0}'.format(self.denom)]))
        else:
            for i, col in enumerate(cols):
                if i == 0:
                    w_output.append(((df[self.denom] * df[col]) + (df['t{0}'.format(self.denom)] * df['w{0}'.format(col)]) + (partial_denom - df[self.denom] - df['t{0}'.format(self.denom)])) / partial_denom)
                else:
                    w_output.append(((df[self.denom] * df[col]) + (df['t{0}'.format(self.denom)] * df['w{0}'.format(col)])) / partial_denom)
        
        return pd.Series(w_output)
        
    def weight_previous(self, year, curr_df):
        prev_year = year - 1
        prev_df = pd.read_csv('projections/{0}_{1}_projections.csv'.format(prev_year, self.stat_type))
        prev_weighted = False
        w_buckets = []
        
        if 'wxWAR' in prev_df.columns:
            prev_weighted = True
        
        if not prev_weighted:
            prev_df['wxWAR'] = prev_df['xWAR']
            prev_df['t{0}'.format(self.denom)] = 0
            
        for bucket in self.war_buckets.keys():
            w_bucket = 'w{0}'.format(bucket)
            w_buckets.append(w_bucket)
            
            if not prev_weighted:
                prev_df[w_bucket] = prev_df[bucket]
        
        prev_df['t{0}'.format(self.denom)] += prev_df[self.denom]
        new_cols = ['playerid', 't{0}'.format(self.denom), 'wxWAR']
        new_cols[2:2] = w_buckets
        prev_df = prev_df[new_cols]
        
        out_df = curr_df.merge(prev_df, on='playerid', how='left')
        out_df.fillna(value=0, inplace=True)
        w_cols = w_buckets
        w_cols.append('wxWAR')
        out_df[w_cols] = out_df.apply(self.historical_weights, axis=1)
        out_df.drop(list(self.war_buckets.keys()), axis=1, inplace=True)
        out_df.drop('xWAR', axis=1, inplace=True)
        out_df.sort_values(by='wxWAR', ascending=False, inplace=True)
    
        return out_df
    
    
    def run_models(self, year, weight_historic=True, verbose=False):
        if verbose:
            print('running {0} {1} projections...'.format(year, self.stat_type))
        projection_input = pd.read_csv('reports/{0}_milb_{1}.csv'.format(year, self.stat_type))
        projection_input = projection_input[projection_input['Age'] < 30]
        projection_input.fillna(value=0, inplace=True)
        projection_input = self.add_exp(projection_input, year)
        projection_input['mlb_team'] = projection_input['Team'].str.split(' ', expand=True)[0]
        projection_output = pd.DataFrame()
        hidden_cols = []
        
        if self.stat_type == 'pit':
            projection_input['O'] = projection_input['IP'].astype(int) * 3 + round(projection_input['IP'] % 1, 1) * 10
            projection_input['O/G'] = projection_input['O'] / projection_input['G']
            projection_input.fillna(value=0, inplace=True)
        
        for level_group in self.level_groups.keys():
            levels = self.level_groups[level_group]
            for level in levels:
                if level not in projection_input.columns:
                    projection_input[level] = 0
                    hidden_cols.append(level)
            level_df = projection_input[projection_input[levels].sum(axis=1) > 0]
            level_mean = self.milb_sample[self.milb_sample[levels].sum(axis=1) > 0]
            
            if self.stat_type == 'bat':
                for position in self.positions:
                    level_df = self.add_pos(level_df, year, position)
                
                level_df.loc[level_df['C'] == 1, '1B'] = 0
                pos_sums = level_df[self.positions].sum(axis=1)
                level_df[self.positions] = level_df[self.positions].div(pos_sums, axis=0)
                level_df.fillna(value=0, inplace=True)
                
            regressed_level = level_df.copy()
            for feature in self.features:
                if feature not in self.static_vars and feature not in self.positions:
                    regressed_level.loc[:, feature] = level_mean[feature].mean()
            
            if len(level_df) > 0:
                level_features = self.features.copy()
                level_features.extend(levels)
                        
                x = level_df[level_features]  
                x_regressed = regressed_level[level_features]
                level_output = level_df.copy()
                level_output[list(self.war_buckets.keys())] = self.models[level_group].predict_proba(x)
                
                regressed_output = ['r{0}'.format(x) for x in list(self.war_buckets.keys())]
                level_output[regressed_output] = self.models[level_group].predict_proba(x_regressed)
                projection_output = projection_output.append(level_output)
        
        curr_output = self.weight_current(projection_output, year)
        for col in hidden_cols:
            if col in curr_output.columns:
                curr_output.drop(col, axis=1, inplace=True)
                
        if weight_historic:
            final_output = self.weight_previous(year, curr_output)
            final_output.to_csv('projections/{0}_{1}_projections.csv'.format(year, self.stat_type), index=False)
        else:
            curr_output.to_csv('projections/{0}_{1}_projections.csv'.format(year, self.stat_type), index=False)
