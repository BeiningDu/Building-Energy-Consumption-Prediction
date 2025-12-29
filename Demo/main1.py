import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV,cross_val_score
from sklearn.metrics import mean_absolute_error,make_scorer
from sklearn.linear_model import Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1.数据加载与清洗
fpath='./building_energy_cleaned.csv'
df=pd.read_csv(fpath,skiprows=1,header=None,names=['timestamp','Power','airTemperature','dewTemperature','windSpeed','hour','day_of_week','month'])
features = ['hour','day_of_week','month','airTemperature','dewTemperature','windSpeed']
X=df[features].copy()
y=df['Power'].copy()
print(f"样本数:{X.shape[0]}")

# 2.随机森林回归器、XGBoost模型调优
print("\nStep1: Optimizing Random Forest Regressor...")
rf_params={
    'n_estimators':[100,200,300],
    'max_depth':[None,10,20,30],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4],
    'max_features':['sqrt','log2',None]
}
# 使用五折交叉验证，使用全部cpu，不显示信息日志（下同）
rf_opt=RandomizedSearchCV(RandomForestRegressor(random_state=42),param_distributions=rf_params,n_iter=30,cv=5,scoring='neg_mean_absolute_error',n_jobs=-1,random_state=42,verbose=0)
rf_opt.fit(X,y)
best_rf=rf_opt.best_estimator_
print(f"RF最佳MAE:{-rf_opt.best_score_:.4f}")
print(f"RF最佳参数:{rf_opt.best_params_}")

# 5.构建未来一周特征
last_ts=pd.to_datetime(df['timestamp'].iloc[-1])
# 生成未来时间序列，提取时间特征
future_idx=pd.date_range(start=last_ts+pd.Timedelta(hours=1),periods=168,freq='H')
future_df=pd.DataFrame({'timestamp':future_idx})
future_df['hour']=future_df['timestamp'].dt.hour
future_df['day_of_week']=future_df['timestamp'].dt.dayofweek
future_df['month']=future_df['timestamp'].dt.month

# 6.模拟气象数据(取最近7天循环)
recent_meteo=df[['airTemperature','dewTemperature','windSpeed']].tail(168)
n_cycles=int(np.ceil(168/len(recent_meteo)))
sim_meteo=pd.concat([recent_meteo]*n_cycles,ignore_index=True).iloc[:168]
X_future=pd.concat([future_df[['hour','day_of_week','month']],sim_meteo.reset_index(drop=True)],axis=1)

# 7.预测与可视化
y_pred=best_rf.predict(X_future)
plt.rcParams['font.sans-serif']=['PingFang SC']
plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(2,1,figsize=(14,10))

# 图1: 历史拟合(考虑圣诞假的误差，取非12月数据的最后14天展示)
y_train_hat=best_rf.predict(X)
sub_df=df[df['month']!=12].iloc[-336:]
ts_history=pd.to_datetime(sub_df['timestamp'])

axes[0].plot(ts_history, y.loc[sub_df.index], label='真实值', c='tab:blue', alpha=0.8)
axes[0].plot(ts_history, y_train_hat[sub_df.index], label='拟合值', c='tab:orange', ls='--')
axes[0].set_title('历史数据拟合效果(RFR)')
axes[0].set_ylabel('能耗(kWh)')
axes[0].legend()
axes[0].grid(True,ls='--',alpha=0.5)

# 图2: 未来72h能耗预测
axes[1].plot(future_df['timestamp'],y_pred,label='预测值',c='tab:red',marker='o',ms=3)
axes[1].set_title('未来一周能耗预测(RFR)')
axes[1].set_ylabel('预测能耗(kWh)')
axes[1].legend()
axes[1].grid(True,ls='--',alpha=0.5)

plt.tight_layout()
plt.show()

# 保存预测结果
res=pd.DataFrame({'timestamp':future_idx,'predicted_power':y_pred})
res.to_csv('future_72h_prediction.csv',index=False)