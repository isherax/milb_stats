import requests
from bs4 import BeautifulSoup

def get_mlb_leaderboard(season, stats, pos='all', lg='all', qual='y', stat_type='8', season1='', ind='0', team='0', 
                        rost='', age='', players='', startdate='', enddate=''):
    url = 'https://www.fangraphs.com/leaders.aspx'
    params = {
        'pos': pos,
        'stats': stats,
        'lg': lg,
        'qual': qual,
        'type': stat_type,
        'season': season,
        'month': '0',
        'season1': season1,
        'ind': ind,
        'team': team,
        'rost': rost,
        'age': age,
        'players': players,
        'startdate': startdate,
        'enddate': enddate,
    }
    if startdate and enddate: params['month'] = '1000'
    
    with requests.session() as session:
        session.headers.update({'user-agent': 'Mozilla/5.0'})
        get = session.get(url, params=params)
        soup = BeautifulSoup(get.content, 'html.parser')
        viewstate  = soup.find('input', {'id': '__VIEWSTATE'})['value']
        validation = soup.find('input', {'id': '__EVENTVALIDATION'})['value']
        post = session.post(url, params=params, data={
                '__EVENTTARGET' : 'LeaderBoard1$cmdCSV', 
                '__VIEWSTATE' : viewstate,
                '__EVENTVALIDATION' : validation,
        })

        return post.content

def get_milb_leaderboard(season, stats, lg, pos='all', qual='y', stat_type='0', ind='', team='0', players=''):
    url = 'https://www.fangraphs.com/minorleaders.aspx'
    params = {
        'pos': pos,
        'stats': stats,
        'lg': lg,
        'qual': qual,
        'type': stat_type,
        'season': season,
        'team': team,
        'players': players,
    }
    
    with requests.session() as session:
        session.headers.update({'user-agent': 'Mozilla/5.0'})
        get = session.get(url, params=params)
        soup = BeautifulSoup(get.content, 'html.parser')
        viewstate  = soup.find('input', {'id': '__VIEWSTATE'})['value']
        validation = soup.find('input', {'id': '__EVENTVALIDATION'})['value']
        if ind: params['team'] += ',to'
        post = session.post(url, params=params, data={
                '__EVENTTARGET' : 'MinorBoard1$cmdCSV', 
                '__VIEWSTATE' : viewstate,
                '__EVENTVALIDATION' : validation,
        })

        return post.content
    
def get_projections(system, stats, pos='all', team='0', lg='all', players='0'):
    url = 'https://www.fangraphs.com/projections.aspx'
    params = {
        'pos': pos,
        'stats': stats,
        'type': system,
        'team': team,
        'lg': lg,
        'players': players,
    }
    
    with requests.session() as session:
        session.headers.update({'user-agent': 'Mozilla/5.0'})
        get = session.get(url, params=params)
        soup = BeautifulSoup(get.content, 'html.parser')
        viewstate  = soup.find('input', {'id': '__VIEWSTATE'})['value']
        validation = soup.find('input', {'id': '__EVENTVALIDATION'})['value']
        post = session.post(url, params=params, data={
                '__EVENTTARGET' : 'ProjectionBoard1$cmdCSV', 
                '__VIEWSTATE' : viewstate,
                '__EVENTVALIDATION' : validation,
        })

        return post.content
