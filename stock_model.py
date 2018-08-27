from IPython import get_ipython
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.layers.core import Dense, Activation, Dropout,Reshape
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional ,RepeatVector,TimeDistributed
from keras.models import load_model
from keras.models import Sequential
import keras
import pandas as pd
from talib import abstract
import talib
import datetime
tf.set_random_seed(777)  # reproducibility
#get_ipython().run_line_magic('matplotlib', 'tk')

class STOCK_model:

    def __init__(self,stock_code,history_dir=None,model_dir=None,predict_dir=None):
        ##parameters
        
        self.windowsize=45        
        self.seq_length = 60
        self.label_seq_length=5
        self.data_dim =23
        self.do_retrain=False
        self.stock_code=stock_code

        self.history_dir=history_dir
        if history_dir==None:
            self.history_dir='./stock/kospi200'
       
        self.model_dir=model_dir
        if model_dir==None:
            self.model_dir='./saved_model'
            
        self.predict_dir=predict_dir    
        if predict_dir==None:
            self.predict_dir='./stock/result200'
            

    

    def normal_windows_minmax(self,window_data):
        ## normalize each window(60*23) and concate
        batch=[]
        for window in window_data:
            normalised_data = []
            for i in range(np.shape(window)[0]):
                feature=[]
                min_array=window[:60].min(axis=0)
                max_array=window[:60].max(axis=0)
                for j in range(np.shape(window)[1]):#data_dim):
                    feature.extend([(window[i][j]-min_array[j])/(max_array[j]-min_array[j])])
                normalised_data.append(feature)
            batch.append(normalised_data)
        return np.array(batch),[min_array,max_array]
          
    def tech_indicators(self,raw_data):
        ##append technical analyze
        np_data=np.array(raw_data)
        inputs={
              'open':np.array(np_data[:,0],dtype='f8'),
              'high':np.array(np_data[:,1],dtype='f8'),
              'low':np.array(np_data[:,2],dtype='f8'),
              'close':np.array(np_data[:,3],dtype='f8'),
              'volume':np.array(np_data[:,4],dtype='f8')
              }
        ##open high low close adj.close volume BBands(h m l),stochastic, (moving average20 60 120) 15 feature
        upper, middle, lower =talib.abstract.BBANDS(inputs, 20, 2, 2)
        ###########
        slowk,slowd=talib.abstract.STOCH(inputs)
        ###########
        dema=talib.abstract.DEMA(inputs)
        ###########    
        ma_40=talib.abstract.MA(inputs,timeperiod=40)
        #ma_200=talib.abstract.MA(inputs,timeperiod=200)
        ma_80=talib.abstract.MA(inputs,timeperiod=80)
        ma_120=talib.abstract.MA(inputs,timeperiod=120)
        ###########
        macda,macdb,macdc=talib.abstract.MACD(inputs)
        ###########
        adx = talib.abstract.ADX(inputs)
        ###########
        trix=talib.abstract.TRIX(inputs)
        ###########
        obv=talib.abstract.OBV(inputs)
        #ad=talib.abstract.AD(inputs)
        ###########
        sar=talib.abstract.SAR(inputs)
        ###########
        midpoint=talib.abstract.MIDPOINT(inputs)   
    
        for i,data in enumerate(raw_data):
            data.extend([upper[i],middle[i],lower[i],slowk[i],slowd[i],dema[i],ma_40[i],ma_80[i],ma_120[i]])#,ma_200[i]
            data.extend([macda[i],macdb[i],macdc[i],adx[i],trix[i],obv[i],sar[i],midpoint[i]])
    
        return np.array(raw_data)
    
    def read_file(self):
        #get date
        date = pd.read_csv(os.path.join(self.history_dir,self.stock_code), index_col=None ,parse_dates=True, usecols=['date'])
        date=date.values.tolist()
        #read history data and use last 2200 days data 
        df = pd.read_csv(os.path.join(self.history_dir,self.stock_code), parse_dates=True, usecols=['open','high','low','close','adjclose','volume'], na_values=['nan'])
        df = df.tail(2200)
        df=df.fillna(method='ffill')
        data=df.values.tolist()

        #add technical indicators
        raw_data=self.tech_indicators(data)
        #if history is too short
        if np.shape(raw_data)[0] < 252:  
            raise Exception("Too small data for prediction :only {:d} days ".format(np.shape(raw_data)[0]))
        
        #remove first 120 days becuse of moving evrage(120) 
        raw_data=raw_data[120:]
        
        result=[]
        for index in range(len(raw_data) - self.seq_length - self.label_seq_length):
            result.append(raw_data[index: index + self.seq_length+self.label_seq_length])            
        result = np.array(result)
        
        #normalize window
        result,_=self.normal_windows_minmax(result)
        # if re training then use last 500 days only
        if self.do_retrain == True:
            result=result[-500:]
            
        train = result
        np.random.shuffle(train)
        x_train = train[:, :-self.label_seq_length]
        y_train = train[:,-self.label_seq_length:,:4]
        
        return x_train, y_train#, x_test, y_test]
    
    def read_file_for_test(self):
        
           
        date = pd.read_csv(os.path.join(self.history_dir,self.stock_code), index_col=None,parse_dates=True, usecols=['date'])
        date=date.values.tolist()
        
        df = pd.read_csv(os.path.join(self.history_dir,self.stock_code), parse_dates=True, usecols=['open','high','low','close','adjclose','volume'], na_values=['nan'])
        df = df.tail(400)
        df=df.fillna(method='ffill')
        data=df.values.tolist()
        
        raw_data=self.tech_indicators(data)
        ##get last 60days' window
        
        raw_data=raw_data[-60:]           

        x_test,min_max=self.normal_windows_minmax([raw_data])

        return x_test,min_max,date
        
    def predict_model(self):
        ########
        ###load previous model !!### 
        if os.path.exists(os.path.join(self.model_dir,self.stock_code)+'.h5'):
            print("Model already exist!! Load previous model")
            model= load_model(os.path.join(self.model_dir,self.stock_code)+'.h5')
            self.do_retrain=True
            return model

        ####### 512 / 256 biLSTM model 
        input_shape=(self.seq_length, self.data_dim) 
        model = Sequential()
        model.add(Bidirectional(LSTM(512,return_sequences=False),input_shape=input_shape)) 
        model.add(RepeatVector(5))
        model.add(Bidirectional(LSTM(256,  return_sequences=True)))  
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(output_dim=4)))
        model.add(Activation('linear'))
        model.summary()
        
        return model

    def train_once(self, model,X_train, y_train):
        ##re training 
        if self.do_retrain == True:
            adam=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
            model.compile(loss='mse', optimizer=adam)
            model.fit(X_train,y_train,batch_size=1024, nb_epoch=40,validation_split=0.0)#,callbacks=[board])
       # board=keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=0, batch_size=256, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
       
        #first training
        else:
            adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0) 
            model.compile(loss='mse', optimizer=adam)#, metrics=['mse'])
            model.fit(X_train,y_train,batch_size=1024,nb_epoch=400,validation_split=0.15)#,callbacks=[board])
    
    def save_predition_as_csv(self, model, X_test,min_max,date):
        prediction=model.predict(X_test)[0]

        #get next 5 days dates (yy-mm-dd,yy-mm-dd+1,...)        
        after_5days=[]
        for i in range(1,8):
            last_date=datetime.datetime.strptime(date[-1][0],"%Y-%m-%d")
            predict_date=last_date+datetime.timedelta(days=i)
            if predict_date.weekday() != 5 and predict_date.weekday()!=6:
                after_5days.append(predict_date.strftime("%Y-%m-%d"))
        after_5days=np.array(after_5days[:5])    
        
        # reverse normalize to get real data 
        result=[]
        for i, value in enumerate(prediction):
            result.append(value*(min_max[1][:4]-min_max[0][:4]) + min_max[0][:4])
        
        #save
        result=np.round(result)
        final=np.column_stack((after_5days,result))
        filename=os.path.join(self.predict_dir,self.stock_code[:-4]+'-'+date[-1][0])
        print("Saved predict values : ",filename)
        print(result)
        np.savetxt(filename+'.csv', final,comments='',delimiter=',',fmt="%s",header="date,open,high,low,close")
        
        
    def model_save(self,model):
        file_name=self.stock_code+'.h5'
        print("save model :",os.path.join(self.model_dir,file_name))
        model.save(os.path.join(self.model_dir,file_name))
    
    
    def model_load(self): 
        print("load model :",os.path.join(self.model_dir,self.stock_code)+'.h5')
        model= load_model(os.path.join(self.model_dir,self.stock_code)+'.h5')
        return model

if __name__== '__main__':
    stock_file=os.listdir('./stock/kospi200')[1]
    print(stock_file,"is opened")
    
    model=STOCK_model(stock_file)
    lstm_model=model.predict_model()
    X_train, y_train =model.read_file()
    model.train_once(lstm_model,X_train, y_train)
    model.model_save(lstm_model)
    lstm_model=model.model_load()
    x_test,min_max,date=model.read_file_for_test()
    model.save_predition_as_csv(lstm_model,x_test,min_max,date)
