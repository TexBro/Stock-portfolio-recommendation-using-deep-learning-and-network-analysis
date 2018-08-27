# -*- coding: utf-8 -*-
"""
Created on Thu May 10 12:34:53 2018

@author: Donghyun Kim
"""
import pandas as pd

class YahooDailyReader():
    def __init__(self, symbol=None, start=None, end=None):
        import datetime, time
        self.symbol = symbol
        
        # initialize start/end dates if not provided
        if end is None:
            end = datetime.datetime(2018,6,25)
        if start is None:
            start = datetime.datetime(2000,1,1)
        
        self.start = start
        self.end = end
        
        # convert dates to unix time strings
        unix_start = int(time.mktime(self.start.timetuple()))
        day_end = self.end.replace(hour=23, minute=59, second=59)
        unix_end = int(time.mktime(day_end.timetuple()))
        
        url = 'https://finance.yahoo.com/quote/{}/history?'
        url += 'period1={}&period2={}'
        url += '&filter=history'
        url += '&interval=1d'
        url += '&frequency=1d'
        self.url = url.format(self.symbol, unix_start, unix_end)
        
    def read(self):
        import requests, re, json
        try:       
            r = requests.get(self.url)
        
            ptrn = r'root\.App\.main = (.*?);\n}\(this\)\);'
            txt = re.search(ptrn, r.text, re.DOTALL).group(1)
            jsn = json.loads(txt)

            df = pd.DataFrame(jsn['context']['dispatcher']['stores']['HistoricalPriceStore']['prices'])

            
            #df.insert(0, 'symbol', self.symbol)
            df['date'] = pd.to_datetime(df['date'], unit='s').dt.date
            
            # drop rows that aren't prices
            df = df.dropna(subset=['close'])
            
            df = df[['date', 'high', 'low', 'open', 'close', 'volume', 'adjclose']]
            df = df.set_index('date')
            return df
        except Exception as e:
            print(self.symbol,e )
        
def save_kospi200_history(history_dir,yy,mm,dd):
    import datetime
    kospi_list=pd.read_csv('kospi200.csv',sep='\t',engine='python',header=None) 
    for kospi in kospi_list[:].values:
        ticker=kospi[0][:6]
        ydr = YahooDailyReader(ticker+'.KS',end=datetime.datetime(yy,mm,dd))
        df = ydr.read()
        try:
            df=df.iloc[::-1]
            df.to_csv(history_dir+'/{}.csv'.format(ticker))
            print(datetime.datetime(yy,mm,dd),'{}.csv is saved'.format(ticker))
        except AttributeError:
            print('######'+ticker+'is not avaible######' )
        #time.sleep(0.1)
    
if (__name__=="__main__"):
    save_kospi200_history('./stock/kospi200_2014',2018,6,29)
    