#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras import backend as K
import os
import gc
import pandas as pd

import stock_model
import download_kospi


model_dir='model'
predict_dir='./stock/result'
history_dir='./stock/kospi200'
IVESTMENT_CYCYLE=5

def setup_floders():
    #floders init
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    if not os.path.exists('./stock'):
        os.makedirs('./stock')
        
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
        
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    
def train_kospi200(train_date):
    ##training particural day(year-mm-dd) 
    year=train_date.year
    month=train_date.month
    date=train_date.day
    
    ##get history data from yahoo finance
    download_kospi.save_kospi200_history(history_dir,year,month,date)
    
    stock_list=os.listdir(history_dir)
    
    for i, filename in enumerate(stock_list[:]):
        print(i,'/',len(stock_list),filename,"is opened")
        model=stock_model.STOCK_model(filename,history_dir,model_dir,predict_dir)
        try:
            #make model
            predict_model=model.predict_model()     
            #read and make training data
            X_train, y_train =model.read_file()                    
            #training
            model.train_once(predict_model,X_train, y_train)
            #save model
            model.model_save(predict_model)
            #load model
            loaded_model=model.model_load()
            #make test data
            x_test,min_max,date=model.read_file_for_test()
            #predict next 5 days after particural day(year-mm-dd) 
            model.save_predition_as_csv(loaded_model,x_test,min_max,date)
            
            K.clear_session()
            gc.collect()
            
        except Exception as e:
            print(e)
            
if __name__=="__main__":
    # 주말 , 공휴일을 제외하기 위해 삼성전자의 날짜 데이터를 기준으로 함
    date_df=pd.read_csv('./005930.csv', index_col='date' ,parse_dates=True, usecols=['date'], na_values=['nan'])
    start=date_df.index.get_loc('2016-01-08')
    end = date_df.index.get_loc('2017-12-28')
    #초기 폴더 설정
    setup_floders()
    
    date_index=start
    while(date_index < end ):
        train_date=date_df.index[date_index].date()
        #모델 학슴
        train_kospi200(train_date)

        date_index+=IVESTMENT_CYCYLE
