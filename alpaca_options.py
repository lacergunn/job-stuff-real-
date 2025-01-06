from util import *
import sys, os, time, datetime, random, csv
from datetime import date, timedelta, datetime
from alpaca import *
from alpaca.trading.enums import *
from alpaca.trading.requests import *
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from ta import *
from ta.utils import *
from ta.trend import *
from ta.momentum import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

ticker_dict, tickerSymbols = get_categorical_tickers()

#initializing local time
localtime = time.asctime( time.localtime(time.time()) )
split_localtime=localtime.split()
timeInts=split_localtime[3].split(":")
start=(date.today() - timedelta(days=3650)).strftime('%Y-%m-%d')
end=date.today().strftime('%Y-%m-%d')
closeDate=(date.today() +timedelta(days=4)).strftime("%y%m%d")
#initializing alpaca trading api
trading_client = TradingClient('AKUJB43L7JXBDTFIDIWO', '9SdS2i6LSFrRbS3Oar61oabJ9DssvHIi0jridRyM',paper=False)
time.sleep(0.5)
acct = trading_client.get_account()
acct_config = trading_client.get_account_configurations()
time.sleep(0.5)
clock= trading_client.get_clock()
time.sleep(0.5)
portfolio=trading_client.get_all_positions()
time.sleep(0.5)
orders=trading_client.get_orders()

tomorrow=(date.today() +timedelta(days=1))
nextWeek=(date.today() +timedelta(days=7))
nextFri=0
timedel=0
while nextFri==0:
    day=date.today()+timedelta(days=timedel)
    if day.weekday()!=4:
        timedel+=1
    else:
        nextFri=day
print(nextFri)

def getRandStock(): #chooses random stock from ticker set and returns it
        #return random.choice(random.choice(ticker_dict['all']))
        x="SPY"
        return x

def calcVolatility(data):
    changes=np.array([])
    for i in range(1,len(data)):
        changes=np.append(changes,data[i]/data[i-1])
    vol=np.std(changes)
    return vol

def todayPrice(stock):
      yf.pdr_override()
      df = yf.download(tickers=stock,period='1d',interval='1m')
      data = df['Adj Close']
      return data

def getOptionContracts(stock,type,priceDown=None,priceUp=None):
    underlying_symbols = [stock]

    # specify expiration date range
    now = datetime.now()
    day1 = now + timedelta(days = 1)
    day60 = now + timedelta(days = 60)

    req = GetOptionContractsRequest(
    underlying_symbols = underlying_symbols,                     # specify underlying symbols
    status = AssetStatus.ACTIVE,                                 # specify asset status: active (default)
    expiration_date = None,                                      # specify expiration date (specified date + 1 day range)
    expiration_date_gte = day1.date(),                           # we can pass date object
    expiration_date_lte = day60.strftime(format = "%Y-%m-%d"),   # or string
    root_symbol = None,                                          # specify root symbol
    type = type,                                                # specify option type:
    style = ExerciseStyle.AMERICAN,                              # specify option style: american
    strike_price_gte = None,                                     # specify strike price range
    strike_price_lte = None,                                     # specify strike price range
    limit = 100,                                                 # specify limit
    page_token = None,                                           # specify page
)
    res = trading_client.get_option_contracts(req)
    return res.option_contracts

def optionBuy(symbol,qty,TIF=TimeInForce.DAY):
    req = MarketOrderRequest(
    symbol = symbol,
    qty = qty,
    side = OrderSide.BUY,
    type = OrderType.MARKET,
    time_in_force = TIF
    )
    trading_client.submit_order(req)
    print(symbol,"purchased with qt:",qty)

def OptionSell(symbol,qty,TIF=TimeInForce.DAY):
    req = MarketOrderRequest(
    symbol = symbol,
    qty = qty,
    side = OrderSide.SELL,
    type = OrderType.MARKET,
    time_in_force = TIF
    )
    trading_client.submit_order(req)
    print(symbol,"sold with qt:",qty)


def quickBuy(stock,qty,type):
    prices=get_tick_values(stock,start,end)
    contracts=getOptionContracts([stock],type,round(prices[-1]),round(prices[-1]+25))
    for i in contracts.option_contracts:
        if i.strike_price==round(prices[-1]):
            symbol=i.symbol
    optionBuy(symbol,qty)

def postMarketPredict(stock,start,end,pastHist):
    algo = LSTM_Model(tickerSymbol = stock, start = start, end = end, depth = 0, naive = True,
                   train_test_split = 1.0, past_history = pastHist , values=15)
    today=todayPrice(algo.tickerSymbol)
    algo.get_ticker_values()
    algo.y=np.append(algo.y,today[-1])
    print(algo.y[-5:])
    algo.prepare_test_train()
    algo.model_LSTM()
    algo.infer_future(algo.xtrain, algo.ytrain, algo.tickerSymbol)
    algo.ts=algo.tickerSymbol
    algo.plot_future_values()
    plt.clf()
    print(algo.ytrain[-3:]*algo.training_std+algo.training_mean)
    print(algo.pred_update[0]*algo.training_std+algo.training_mean)
    return algo.pred_update, algo.ytrain

def get_control_vector(val):
    '''
    Returns the mask of day instances where stock purchase/sell decisions are to be made
    :param val: Input array of stock values
    :return: np.array of decisions maks labels (-2/0/2)
    '''
    return np.diff(np.sign(np.diff(val,axis=0)),axis=0)

def mlCalc(stock,start,end):
    baseVol=calcVolatility(get_tick_values('DOW',start,end))
    vol=calcVolatility(get_tick_values(stock,start,end))
    if vol>baseVol*2:
        print("WARNING, Chosen stock is highly volatile, AI subject to innacuraccy")
        preds,real=postMarketPredict(stock,start,end,30)
    if vol<baseVol*0.5:
        preds,real=postMarketPredict(stock,start,end,100)
    if vol>baseVol*0.5 and vol<baseVol*2:
        preds,real=postMarketPredict(stock,start,end,60)
    return get_control_vector(np.append(real[-5:],preds[-1]))

def calcStop(data):
    changes=np.array([])
    for i in range(1,len(data)):
        changes=np.append(changes,np.abs(data[i]/data[i-1]))
    vol=np.average(changes)
    print("Stop trigger is ",(np.abs(1-vol)/2)*100,"%")
    return ((np.abs(1-vol)/2)*100)

def stopMonitor(stock,stop,type):
    '''Live tracker of stock data, checks if a trailing stop is triggered.
        Included because neither alpaca nor robinhood have built in trailing stops for options
        :stock - Stock name in string or char form
        :stop - trailing stop trigger value, must be less than 1 (0.01=1% trailing stop)
        :type - user may want to use a trailing stop for a short, the direction the trailing stop looks depends on the type of purchase. Use ContractType'''
    print("tracking stop loss on  ",stock)
    peak=todayPrice(stock)
    peak=peak[-1]
    print("Trailing stop loss activated")
    if clock.is_open==False:
        print("Market is closed, deactivating stop loss")
        return True
    while clock.is_open:
        current=todayPrice(stock)
        current=current[-1]
        time.sleep(60)
        if type=='C': 
            if current<=peak*(1-stop):
                print("Stop Loss Triggered")
                return False
            else:
                time.sleep(30)
        if type=='P':
            if current>=peak*(1+stop):
                print("Stop Loss Triggered")
                return False
            else:
                time.sleep(30)

def getOptionSymbol(stock,strike,closeDate,Optype):
    '''Turns parameters into option chain symbol.
        :stock - Stock name in string or char form
        :strike - options's strike price as int
        :Optype - char representing call (C) or put (P)
        :closeDate - option's closing date in STRING YYMMDD format'''
    strike=str(int(strike*1000))
    digis=8-len(strike)
    zeroes=""
    for i in range(digis):
        zeroes=zeroes+"0"
    symbol=stock+closeDate+Optype+zeroes+strike
    print(symbol)
    return symbol

def getMaxQty(symbol):
    '''Calculates max amount of option contracts that can be bought at current balance'''
    contract = trading_client.get_option_contract(symbol)
    close=float(contract.close_price)+1
    print(close)
    print(acct.cash)
    max=float(acct.cash)/(close*100)
    print(max)
    return int(max)

def getStock(symbol):
    '''gets stock ticker from option symbol'''
    for i in range(len(symbol)):
        char=symbol[i]
        if char.isdigit():
            stock=symbol.split(char)
            return stock[0]

def getType(symbol):
    '''Returns option trade type (C(call) or P(put))'''
    for i in range(1,len(symbol)):
        char=symbol[-i]
        if char.isdigit()==False:
            return char


def basicOption(stock,strike,type,date):
    list=getOptionContracts(stock,type,date)
    ITM=[]
    OTM=[]
    for i in list:
        if float(i.strike_price)<=strike:
            ITM=np.append(ITM,i)
        if float(i.strike_price)>=strike:
            OTM=np.append(OTM,i)
    if len(ITM)<1:
        print(OTM[0])
        return OTM[0]
    else:
        print(ITM[-1])
        return ITM[-1]



def prepKNN(stock,start,end):
    prices=get_tick_values(stock,start,end)
    today=todayPrice(stock)
    prices=np.append(prices,today[-1])
    directions=[]
    for i in range(1,len(prices)):
        if prices[i]>prices[i-1]:
            directions=np.append(directions,1)
        if prices[i]<=prices[i-1]:
            directions=np.append(directions,0)
    prices=np.delete(prices,-1)
    avgdifs=[]
    relStren=rsi(pd.Series(prices),14,True)
    avgs=sma_indicator(pd.Series(prices),10,True)
    stochK=stochrsi_k(pd.Series(prices),14,3,3,True)
    stochR=stochrsi(pd.Series(prices),14,3,3,True)

    for i in range(len(prices)):
        avgdifs=np.append(avgdifs,prices[i]-avgs[i])
    print(len(avgdifs),len(relStren),len(directions),len(prices),len(stochK),len(stochR))
    data={
        "Moving Averages": avgdifs,
        "Relative Strength Index": relStren,
        "Stochastic": stochK,
        "Stochastic RSI": stochR,
        
        "Directions": directions
    }
    df = pd.DataFrame(data)

    X=df[["Moving Averages","Relative Strength Index","Stochastic","Stochastic RSI"]].values.astype(float)
    Y=df[["Directions"]].values.astype(float)

    X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
    print ('Train set:', X_train.shape,  y_train.shape)
    print ('Test set:', X_test.shape,  y_test.shape)

    Ks = 100
    mean_acc = np.zeros((Ks-1))
    std_acc = np.zeros((Ks-1))

    for n in range(1,Ks):
        
        #Train Model and Predict  
        neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
        yhat=neigh.predict(X_test)
        mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
        std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    print(mean_acc.max())

    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.01, random_state=4)
    KNNmod = KNeighborsClassifier(n_neighbors = mean_acc.argmax()+1).fit(X_train,y_train)
    return KNNmod

def prepKNNWeekly(stock,start,end):
    prices=get_tick_values(stock,start,end)
    pricesTotal=prices
    prices=pricesTotal[0:-1:5]
    print(len(prices))
    directions=[]
    for i in range(1,len(prices)):
        if prices[i]>prices[i-1]:
            directions=np.append(directions,1)
        if prices[i]<=prices[i-1]:
            directions=np.append(directions,0)
    avgdifs=[]
    relStren=rsi(pd.Series(prices),14,True)
    avgs=sma_indicator(pd.Series(prices),10,True)
    stochK=stochrsi_k(pd.Series(prices),14,3,3,True)
    stochR=stochrsi(pd.Series(prices),14,3,3,True)

    for i in range(len(prices)):
        avgdifs=np.append(avgdifs,prices[i]-avgs[i])
    print(len([avgdifs[len(avgdifs)-1]]),len([relStren[len(relStren)-1]]),len(directions),len(prices),len([stochK[len(stochK)-1]]),len([stochR[len(stochR)-1]]))
    data={
        "Moving Averages": avgdifs[0:-1],
        "Relative Strength Index": relStren[0:-1],
        "Stochastic": stochK[0:-1],
        "Stochastic RSI": stochR[0:-1],
        
        "Directions": directions
    }
    df = pd.DataFrame(data)

    X=df[["Moving Averages","Relative Strength Index","Stochastic","Stochastic RSI"]].values.astype(float)
    Y=df[["Directions"]].values.astype(float)

    X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
    print ('Train set:', X_train.shape,  y_train.shape)
    print ('Test set:', X_test.shape,  y_test.shape)

    Ks = 100
    mean_acc = np.zeros((Ks-1))
    std_acc = np.zeros((Ks-1))

    for n in range(1,Ks):
        
        #Train Model and Predict  
        neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
        yhat=neigh.predict(X_test)
        mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
        std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    print(mean_acc.max())

    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.01, random_state=4)
    KNNmod = KNeighborsClassifier(n_neighbors = mean_acc.argmax()+1).fit(X_train,y_train)
    return KNNmod



def KNNCalc(stock,model):
    prices=get_tick_values(stock,start,end)
    today=todayPrice(stock)
    prices=np.append(prices,today[-1])
    avgdifs=[]
    if prices[0]==prices[-1]:
        prices=np.delete(prices,-1)
    print(prices[-1])
    relStren=rsi(pd.Series(prices),14,True)
    avgs=sma_indicator(pd.Series(prices),10,True)
    stochK=stochrsi_k(pd.Series(prices),14,3,3,True)
    stochR=stochrsi(pd.Series(prices),14,3,3,True)
    
    #print(avgdifs,relStren,stochK,stochR)

    for i in range(len(prices)):
        avgdifs=np.append(avgdifs,prices[i]-avgs[i])
    print(len(avgdifs),len(relStren),len(prices),len(stochK),len(stochR))
    data={
        "Moving Averages": [avgdifs[len(avgdifs)-1]],
        "Relative Strength Index": [relStren[len(relStren)-1]],
        "Stochastic": [stochK[len(stochK)-1]],
        "Stochastic RSI": [stochR[len(stochR)-1]],
    }
    df = pd.DataFrame(data)

    X=df[["Moving Averages","Relative Strength Index","Stochastic","Stochastic RSI"]].values.astype(float)

    yhat=model.predict(X)
    return yhat

def closestMultiple(n, x): 
    if x > n: 
        return x; 
    z = (int)(x / 2); 
    n = n + z; 
    n = n - (n % x); 
    return n; 
startValue=float(acct.portfolio_value)
time.sleep(0.5)
PremarketComplete=False
MarketComplete=False
chosen=None
choices=[]

while 1==1:
    open=clock.is_open
    localtime = time.asctime( time.localtime(time.time()))
    time.sleep(10)
    while open==False:
        
        split_localtime=localtime.split()
        timeInts=split_localtime[3].split(":")
        if (int(timeInts[0])>16 and int(timeInts[0])<24) and PremarketComplete==True:
            print('Market is closed, come back later')
            print(localtime)
            PremarketComplete==False
        elif (int(timeInts[0])<9 or  int(timeInts[0])>=22) and PremarketComplete==False:
            if len(portfolio)>0 and len(orders)==1:
                print('Stocks and Orders in place, have a nice day')
                PremarketComplete=True
            elif len(portfolio)==0 and len(orders)==1:
                print('Orders in place, have a nice day')
                PremarketComplete=True
                
        elif PremarketComplete==False and len(portfolio)==0 and len(trading_client.get_orders())==0:
            time.sleep(1)
            print('No stocks or pending purchases')
            print(localtime)
            choice=getRandStock()
            if choice not in choices:    
                choices.append(choice)
                print(choice)
                model=prepKNNWeekly(choice,start,end)
                control=KNNCalc(choice,model)
                print(control)
                if control[-1]==1:
                    chosen=choice
                    print('Chosen stock is '+chosen)
                    data=get_tick_values(chosen,date.today() - timedelta(days=10),date.today().strftime('%Y-%m-%d'))
                    price = data[-1]
                    chosen = basicOption(chosen,price,"call",nextFri)
                    PremarketComplete=True
                if control[-1]==0:
                    chosen=choice
                    print('Shorting '+chosen)
                    data=get_tick_values(chosen,date.today() - timedelta(days=10),date.today().strftime('%Y-%m-%d'))
                    price = data[-1]
                    chosen = basicOption(chosen,price,"put",nextFri)
                    PremarketComplete=True
         
    while open==True and MarketComplete==False:
        print("Market open, ding ding ding!")
        split_localtime=localtime.split()
        timeInts=split_localtime[3].split(":")
        if chosen!=None:
            data=get_tick_values(getStock(chosen),date.today() - timedelta(days=10),date.today().strftime('%Y-%m-%d'))
            price = int(data[-1])
            maxshares=getMaxQty(chosen.symbol)-1
            time.sleep(0.5)
            optionBuy(chosen.symbol,maxshares)
            print("Buying ",maxshares," contracts of ",chosen.symbol," at strike price ",chosen.strike_price)
            MarketComplete=True
        elif len(trading_client.get_orders())==0 and len(portfolio)>0 and (int(timeInts[0])>=9 or  int(timeInts[0])<=3):
            MarketComplete=True
            model=prepKNNWeekly(getStock(portfolio[0].symbol),start,end)
            control=KNNCalc(getStock(portfolio[0].symbol),model)
            print(control)
            if control==1 and getType(portfolio[0].symbol)=='P':
                OptionSell(portfolio[0].symbol,portfolio[0].qty)
            if control==0 and getType(portfolio[0].symbol)=='C':
                OptionSell(portfolio[0].symbol,portfolio[0].qty)
            else:
                data=get_tick_values(getStock(portfolio[0].symbol),date.today() - timedelta(days=30),date.today().strftime('%Y-%m-%d'))
                stop=True
                stop=stopMonitor(getStock(portfolio[0].symbol),calcStop(data,5),getType(portfolio[0].symbol))
                MarketComplete=True
                if stop==False:
                    OptionSell(portfolio[0].symbol,portfolio[0].qty)
            
        elif PremarketComplete==False and len(portfolio)==0:
            print('Running initial stock sims because SOMEONE forgot to.')
            print('Be warned: Analysis is designed with start of day purchases in mind, profits are even less garunteed than normal.')
            print(localtime)
            choice=getRandStock()
            print(choice)
            model=prepKNNWeekly(choice,start,end)
            control=KNNCalc(choice,model)
            print(control)
            if control[-1]==1:
                chosen=choice
                print('Chosen stock is '+chosen)
                data=get_tick_values(chosen,date.today() - timedelta(days=30),date.today().strftime('%Y-%m-%d'))
                price = int(data[-1])
                chosen=basicOption(chosen,price,"call",nextFri)
                maxshares=getMaxQty(chosen.symbol)-3
                time.sleep(0.5)
                optionBuy(chosen.symbol,maxshares)
                time.sleep(5)
                PremarketComplete=True
                MarketComplete=True
                stop=True
                stop=stopMonitor(choice,calcStop(data,5),'C')
                if stop==False:
                    OptionSell(portfolio[0].symbol,portfolio[0].qty)
            if control[-1]==0:
                chosen=choice
                print('Chosen stock is '+chosen)
                data=get_tick_values(chosen,date.today() - timedelta(days=30),date.today().strftime('%Y-%m-%d'))
                price = int(data[-1]) 
                chosen=basicOption(chosen,price,"put",nextFri)
                maxshares=getMaxQty(chosen.symbol)-3
                time.sleep(0.5)
                optionBuy(chosen.symbol,maxshares)
                time.sleep(5)
                PremarketComplete=True
                MarketComplete=True
                stop=True
                stop=stopMonitor(choice,calcStop(data,5 ),'P')
                if stop==False:
                    OptionSell(portfolio[0].symbol,portfolio[0].qty)