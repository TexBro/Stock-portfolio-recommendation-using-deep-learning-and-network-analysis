# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 23:41:32 2018

@author: Donghyun Kim
"""

import math
import numpy as np
import pandas as pd
import os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import random
from pyclustering.cluster.kmedoids import kmedoids

STOCK_PATH='./stock/kospi200'
PREDICTION_PATH='./stock/result200'
Is_chosen_by_predict=True


def get_predict(index_labels,one_edge_stocks,Predict_date):
    
    return_rate=[]
    low_start_return_rate=[]
    under_stocks=[]
    for i in one_edge_stocks:
        #real value
        df_y = pd.read_csv(STOCK_PATH+'/'+index_labels[i], index_col='date',parse_dates=True, usecols=['date', 'close'], na_values=['nan'])
        y=df_y.loc[Predict_date].values[0]

        #predict value 
        df_pred = pd.read_csv(PREDICTION_PATH+'/'+index_labels[i][:-4]+'-'+Predict_date+'.csv',index_col='date',parse_dates=True, usecols=['date', 'close'], na_values=['nan'])
        #누적 수익률 계산
        pred_daily_return = df_pred['close']/df_pred.shift(1)['close']
        pred_return_sum = pred_daily_return[1:].values.sum()
        
        next_day_prediction=df_pred['close'][0] #1 day after
        
        #원래 가격보다 낮은 위치에서 예측이 시작된 종목들
        if next_day_prediction-y <= 0:
            low_start_return_rate.append(pred_return_sum)
            under_stocks.append(i)
            
        return_rate.append(pred_return_sum)

    if len(under_stocks)==0:
        stock=one_edge_stocks[return_rate.index(max(return_rate))]
    else:   
        stock=under_stocks[low_start_return_rate.index(max(low_start_return_rate))]

    return stock

def init_portfolio(num_assets, df_daily_returns):
	global noa
	global daily_returns
	noa = num_assets
	daily_returns = df_daily_returns

def get_data(symbols,B_date):
    df = pd.DataFrame()
    # 과거 데이터 120일 중 누락된지 여부 확인
    standard_df=pd.read_csv('./005930.csv', index_col='date' ,parse_dates=True, usecols=['date', 'adjclose'], na_values=['nan'])
    begin=standard_df.index.get_loc(B_date)
    standard_df=standard_df[begin-120:begin]
    stnadard_date=standard_df.index[0]
    
    for symbol in symbols:
        try:
            df_temp = pd.read_csv(STOCK_PATH+'/'+symbol, index_col='date' ,parse_dates=True, usecols=['date', 'adjclose'], na_values=['nan'])
            df_temp = df_temp.fillna(method='ffill')
            begin=df_temp.index.get_loc(B_date)
            df_temp=df_temp[begin-120:begin]
            df_temp = df_temp.rename(columns={'adjclose': symbol})
            #120일치가 다 있는 경우
            if (df_temp.index[0] == stnadard_date):
                df = df.join(df_temp, how='outer')
            else:
                #print("missing : ",symbol)
                pass
                
        except Exception as e:
            print(e)
            pass  
        
    return df

def get_daily_returns(df):
	daily_returns = np.log(df/df.shift(1))
	daily_returns = daily_returns[1:]
	return daily_returns

def get_MST(distnace_matrix):
    X=csr_matrix(distnace_matrix)
    # =X.toarray().astype(float)
    Tcsr=minimum_spanning_tree(X)
    MST=Tcsr.toarray().astype(float)
    return MST

def get_K_medoid(distnace_matrix):
    kmedoids_instance=kmedoids(distnace_matrix,[random.randrange(0,noa) for i in range(10)],data_type='distance_matrix')
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    medoids = kmedoids_instance.get_medoids()
    return clusters,medoids

def count_degree(np_MST_matrix):
    degree_count=[]
    
    for i in range(noa):
        line=np.count_nonzero(np_MST_matrix[i])
        line+=np.count_nonzero(np_MST_matrix[:,i])
        degree_count.append(line)
        
    return degree_count
    
def get_one_degree_nodes(clusters,degree_num):
    one_degree_in_each_cluster=[]
    
    for cluster in clusters:
        one_edge=[]
        for i in cluster:
            if degree_num[i] == 1:
                one_edge.append(i)
        one_degree_in_each_cluster.append(one_edge)
    return one_degree_in_each_cluster     

def calculate_profit(final_10_stocks,Buy_date,Sell_date,index_labels):

    total_buy=0
    total_sell=0
    
    for stock_num in final_10_stocks:

        df_temp = pd.read_csv(STOCK_PATH +'/'+index_labels[stock_num], index_col='date',na_values=['nan'])
        df_temp.fillna(method='ffill')
        buy_price = df_temp['open'][df_temp.index.get_loc(Buy_date)]
        sell_price=df_temp['close'][df_temp.index.get_loc(Sell_date)]
        total_buy+=buy_price
        total_sell+=sell_price
    
    return round(((total_sell-total_buy)/total_sell),4)

if (__name__ == "__main__"):
    ###에측값이 없는 종목들은 삭제 필요


    date_df=pd.read_csv('./005930.csv', index_col='date' ,parse_dates=True, usecols=['date'], na_values=['nan'])
    start=date_df.index.get_loc('2015-01-05')
    end = date_df.index.get_loc('2015-08-03')

    accumulate_rate=1
    date_index=start
    while(date_index < end):
        
        ##시나리오 : predict 하루동안 훈련시켜 하루 후인 buy date open 가격에 사서 sell date close 가격에 팜
        datetime_date=date_df.index[date_index].date()
        Predict_date=date_df.index[date_index].strftime('%Y-%m-%d')
        Buy_date=date_df.index[date_index+1].strftime('%Y-%m-%d')
        date_index+=5
        Sell_date=date_df.index[date_index].strftime('%Y-%m-%d')
        date_index+=1
        
        profit_total=0
        count=1    
        while(count<=1):
            try:
                stocklist=os.listdir(STOCK_PATH)
                
                adj_close = get_data(stocklist,Predict_date)               
                daily_returns = get_daily_returns(adj_close)
            
                init_portfolio(daily_returns.shape[1], daily_returns)
                
                corr=daily_returns.corr(method='pearson')
                
                distance= np.sqrt(2*(1-corr))
                
                distnace_matrix=distance.values.tolist()
                
                index_labels=distance.index.values.tolist()
                
                ## MST 
                MST_matrix=get_MST(distnace_matrix)
                ## K -medoid
                clusters,mediods=get_K_medoid(distnace_matrix)

                '''
                #draw MST                             
                colors = [(random.random(), random.random(), random.random()) for _i in range(noa)]                
                G = nx.Graph(MST_matrix)                
                nx.draw(G,node_color=colors ,with_labels=True)
                '''
                
                np_MST_matrix=np.array(MST_matrix)
                degree_num=count_degree(np_MST_matrix)
                one_degree_in_each_cluster=get_one_degree_nodes(clusters,degree_num)
                        
                final_10_stocks=[]
                for i in one_degree_in_each_cluster:
                    
                    if Is_chosen_by_predict:
                        final_10_stocks.append(get_predict(index_labels,i,Predict_date))
                    else:
                        final_10_stocks.append(random.choice(i))

                final_10_stocks.sort()

                profit_total+=calculate_profit(final_10_stocks,Buy_date,Sell_date,index_labels)

                count+=1
                
            except Exception as e:
                print(e)
                pass
            
        print(Buy_date," ~ ",Sell_date," : ",round(profit_total/(count-1)+1,4))
        accumulate_rate*=round(profit_total/(count-1)+1,4)
        
    print("total accumlated rate = ",accumulate_rate)