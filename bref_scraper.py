from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import pandas as pd
import unidecode


def get_start_year(name, playerid, init_year):
    name = unidecode.unidecode(name)
    url = 'https://www.baseball-reference.com/search/search.fcgi?search='
    query = '+'.join(name.split())
    new_url = url + query
    print(query)
    
    try:
        soup = BeautifulSoup(urlopen(new_url), features='lxml')
        
        if 'sa' in str(playerid):
            id_type = 'milb_players'
        else:
            id_type = 'players'
            
        query = soup.find_all('div', {'id': id_type})
        results = re.findall('\(.*?\)', str(query))
        
        if len(results) >= 1 and id_type == 'milb_players':
            possible = []
            
            for result in results:
                result = str(result)
                years = result[result.find('(')+1:result.find(')')]
                year = int(years.split('-')[0])
                
                if year >= init_year - 10 and year <= init_year:
                    possible.append(year)
                    
            if len(possible) == 1:
                return possible[0]
                
        else:
            if id_type == 'milb_players':
                query = soup.find_all('tr', {'class': 'minors full'})
                partial_query = soup.find_all('tr', {'class': 'minors partial_table'})
                
                result = re.findall('year=.*?\"', str(query))
                if len(result) > 0:
                    result = int(re.sub('[^0-9]','', result[0]))
                else:
                    result = None
            
                partial_result = re.findall('year=.*?\"', str(partial_query))
                if len(partial_result) > 0:
                    partial_result = int(re.sub('[^0-9]','', partial_result[0]))
                else:
                    partial_result = None
            else:
                query = soup.find_all('tr', {'class': 'minors_table hidden'})
                partial_result = None
                
                result = re.findall('year_ID.*?<', str(query))
                if len(result) > 0:
                    result = int(re.sub('[^0-9]','', result[0]))
                else:
                    result = None
            
            if result is None and partial_result is None:
                return None
            elif result and partial_result is None:
                return result
            elif result is None and partial_result:
                return partial_result
            else:
                return min(result, partial_result)
            
    except Exception as e:
        print(e)
            
    return None
         

if __name__ == '__main__':
    year = 2008
    stats = 'bat'
    df = pd.read_csv('reports/{0}_milb_{1}.csv'.format(year, stats))
    df = df[df['PA'] >= 200]
    df = df[['Name', 'playerid']].drop_duplicates()
    df['start_year'] = df.apply(lambda x: get_start_year(x['Name'], x['playerid'], year), axis=1)
    df.to_csv('reports/exp_data/{0}_{1}_auto.csv'.format(year, stats), index=False)
    