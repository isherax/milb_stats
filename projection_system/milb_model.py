import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class milb_model:
    def __init__(self, stat_type, features, years=None, levels=None, war_buckets=None):
        self.stat_type = stat_type
        self.features = features
        self.models = {}
        self.bucket_means = {}
        self.regressor = pd.DataFrame()
        self.flat_levels = []
        self.positions = ['C', '1B', '2B', 'SS', '3B', 'LF', 'CF', 'RF']
        
        if stat_type == 'bat':
            self.denom = 'PA'
            self.features.extend(self.positions)
        elif stat_type == 'pit':
            self.denom = 'TBF'
        
        if years:
            self.years = years
        else:
            self.years = range(2006, 2013)
            
        if levels:
            self.level_groups = levels
        else:
            self.level_groups = {'high': ['Level_AAA', 'Level_AA'], 'mid': ['Level_A+', 'Level_A'], 'low': ['Level_R-', 'Level_DSL']}
            
        for key in self.level_groups.keys():
            self.flat_levels.extend(self.level_groups[key])
            
        if war_buckets:
            self.war_buckets = war_buckets
        else:
            # self.war_buckets = {'-1': None, '0': [None, 0.5], '1-5': [0.5, 4.5], '5-10': [4.5, 9.5], '10-15': [9.5, 14.5], '15+': [14.5, None]}
            self.war_buckets = {'-1': None, '0': [None, 0.5], '1': [0.5, 1.5], '2': [1.5, 2.5], '3': [2.5, 3.5], '4+': [3.5, None]}
            
            
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
                
        return '-1'
    
    
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
        
    def train_models(self, verbose=False):
        milb_sample = pd.DataFrame()
        
        for year in self.years:
            year_sample = pd.read_csv('reports/{0}_milb_{1}.csv'.format(year, self.stat_type))
            year_sample['Year'] = year
            
            if self.stat_type == 'bat':
                for position in self.positions:
                    year_sample = self.add_pos(year_sample, year, position)
                
            milb_sample = milb_sample.append(year_sample)
        
        mlb_sample = pd.read_csv('peak_mlb_{0}.csv'.format(self.stat_type), dtype={'playerid': str})
        mlb_sample = mlb_sample[['WAR', 'playerid']]
        
        milb_sample = milb_sample[milb_sample[self.denom] >= 200]
        milb_sample = milb_sample[milb_sample['Age'] < 30]
        milb_sample = pd.merge(milb_sample, mlb_sample, on='playerid', how='left')
        milb_sample['WAR_bucket'] = milb_sample.apply(self.create_buckets, axis=1)
        milb_sample.to_csv('test.csv')
        milb_sample.fillna(value=0, inplace=True)
        
        if self.stat_type == 'pit':
            milb_sample['O'] = milb_sample['IP'].astype(int) * 3 + round(milb_sample['IP'] % 1, 1) * 10
            milb_sample['O/G'] = milb_sample['O'] / milb_sample['G']
        
        for key in self.war_buckets.keys():
            mean = milb_sample[milb_sample['WAR_bucket'] == key]['WAR'].mean()
            
            if mean > 0:
                self.bucket_means[key] = mean
            else:
                self.bucket_means[key] = 0
    
        for level_group in self.level_groups.keys():
            levels = self.level_groups[level_group]
            level_sample = milb_sample[milb_sample[levels].sum(axis=1) > 0]
            x = level_sample[self.features]
            y = level_sample['WAR_bucket']
            
            model = LogisticRegression(solver='newton-cg').fit(x, y)
            if verbose:
                print('{0} model score: {1}'.format(level_group, round(model.score(x, y), 3)))
            self.models[level_group] = model
    
    
    def weight_current(self, model_results):
        model_results = model_results[model_results[self.denom] > 0]
        new_cols = self.features
        new_cols.extend(self.flat_levels)
        
        weighted_mean = lambda x: np.average(x, weights=model_results.loc[x.index, self.denom])
        aggregate = dict.fromkeys(new_cols, weighted_mean)
        aggregate[self.denom] = np.sum
        
        for key in self.war_buckets.keys():
            aggregate[key] = weighted_mean
        
        grouped_df = model_results.groupby(['Name', 'playerid']).agg(aggregate)
        out_df = pd.DataFrame(grouped_df)
        out_df['xWAR'] = out_df.apply(self.calc_xwar, axis=1)
        out_df = out_df[out_df[self.denom] >= 100].reset_index()
        out_df.sort_values(by='xWAR', ascending=False, inplace=True)
        
        return out_df
        
        
    def historical_weights(self, df):    
        if self.stat_type == 'bat':
            denom_min = 400
        else:
            denom_min = 200
        cols = list(self.war_buckets.keys())
        cols.append('xWAR')
        w_output = []
        
        if df[self.denom] >= denom_min:
            for col in cols:
                w_output.append(df[col])
        elif df[self.denom] + df['t{0}'.format(self.denom)] >= denom_min:
            for col in cols:
                w_output.append((df[self.denom] * df[col] + (denom_min - df[self.denom]) * df['w{0}'.format(col)]) / denom_min)
        else:
            for i, col in enumerate(cols):
                if i == 0:
                    w_output.append((df[self.denom] * df[col] + df['t{0}'.format(self.denom)] * df['w{0}'.format(col)] + (denom_min - df[self.denom] - df['t{0}'.format(self.denom)])) / denom_min)
                else:
                    w_output.append((df[self.denom] * df[col] + df['t{0}'.format(self.denom)] * df['w{0}'.format(col)]) / denom_min)
        
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
    
    
    def run_models(self, year, weight_historic=True):
        print('running {0} projections...'.format(year))
        projection_input = pd.read_csv('reports/{0}_milb_{1}.csv'.format(year, self.stat_type))
        projection_input = projection_input[projection_input['Age'] < 30]
        projection_input.fillna(value=0, inplace=True)
        projection_output = pd.DataFrame()
        
        if self.stat_type == 'pit':
            projection_input['O'] = projection_input['IP'].astype(int) * 3 + round(projection_input['IP'] % 1, 1) * 10
            projection_input['O/G'] = projection_input['O'] / projection_input['G']
            projection_input.fillna(value=0, inplace=True)
        
        for level_group in self.level_groups.keys():
            levels = self.level_groups[level_group]
            level_df = projection_input[projection_input[levels].sum(axis=1) > 0]
            
            if self.stat_type == 'bat':
                for position in self.positions:
                    level_df = self.add_pos(level_df, year, position)
            
            if len(level_df) > 0:
                x = level_df[self.features]                
                level_output = level_df.copy()
                level_output[list(self.war_buckets.keys())] = self.models[level_group].predict_proba(x)
                projection_output = projection_output.append(level_output)
        
        curr_output = self.weight_current(projection_output)
        if weight_historic:
            final_output = self.weight_previous(year, curr_output)
            final_output.to_csv('projections/{0}_{1}_projections.csv'.format(year, self.stat_type), index=False)
        else:
            curr_output.to_csv('projections/{0}_{1}_projections.csv'.format(year, self.stat_type), index=False)
