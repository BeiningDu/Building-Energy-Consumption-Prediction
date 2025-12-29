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

print("\nStep2: Optimizing XGBoost...")
xgb_params={
    'n_estimators':[200,300,500],
    'learning_rate':[0.01,0.05,0.1,0.2],
    'max_depth':[4,6,8,10],
    'subsample':[0.8,0.9,1.0],
    'colsample_bytree':[0.8,0.9,1.0],
    'gamma':[0,0.1,0.2],
    'reg_alpha':[0,0.1,1],
    'reg_lambda':[1,1.5,2]
}
xgb_opt=RandomizedSearchCV(XGBRegressor(random_state=42,objective='reg:squarederror'),param_distributions=xgb_params,n_iter=40,cv=5,scoring='neg_mean_absolute_error',n_jobs=-1,random_state=42,verbose=0)
xgb_opt.fit(X,y)
best_xgb=xgb_opt.best_estimator_
print(f"XGBoost最佳MAE:{-xgb_opt.best_score_:.4f}")
print(f"XGBoost最佳参数:{xgb_opt.best_params_}")

# 3.其他模型调优
print("\nStep3: Optimizing Ridge...")
ridge = Ridge(random_state=42)
ridge_params = {'alpha':np.logspace(-2,3,20)}
ridge_opt = RandomizedSearchCV(ridge,ridge_params,n_iter=20,cv=5,scoring='neg_mean_absolute_error',n_jobs=-1,random_state=42)
ridge_opt.fit(X,y)
best_ridge = ridge_opt.best_estimator_
print(f"Ridge最佳MAE:{-ridge_opt.best_score_:.4f}")

print("Step4: Optimizing Lasso...")
lasso = Lasso(random_state=42, max_iter=5000)
lasso_params = {'alpha':np.logspace(-4,1,30)}
lasso_opt = RandomizedSearchCV(lasso,lasso_params,n_iter=30,cv=5,scoring='neg_mean_absolute_error',n_jobs=-1,random_state=42)
lasso_opt.fit(X,y)
best_lasso = lasso_opt.best_estimator_
print(f"Lasso最佳MAE:{-lasso_opt.best_score_:.4f}")

print("Step5: Optimizing MLPRegressor...")
mlp_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(random_state=42,max_iter=1000,early_stopping=True,validation_fraction=0.1))
])
mlp_params = {
    'mlp__hidden_layer_sizes': [(50,),(100,),(50,50),(100,50)],
    'mlp__alpha': [0.0001,0.001,0.01,0.1],
    'mlp__learning_rate_init': [0.001,0.01],
    'mlp__activation': ['relu','tanh']
}
mlp_opt = RandomizedSearchCV(mlp_pipe, mlp_params, n_iter=30, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42)
mlp_opt.fit(X, y)
best_mlp = mlp_opt.best_estimator_
print(f"MLP最佳MAE:{-mlp_opt.best_score_:.4f}")

print("Step6: Optimizing KNN...")
knn = KNeighborsRegressor()
knn_params = {'n_neighbors':range(3,31),'weights': ['uniform','distance']}
knn_opt = RandomizedSearchCV(knn,knn_params,n_iter=30,cv=5,scoring='neg_mean_absolute_error',n_jobs=-1,random_state=42)
knn_opt.fit(X,y)
best_knn = knn_opt.best_estimator_
print(f"KNN最佳MAE:{-knn_opt.best_score_:.4f}")

# 4.Stacking融合数个模型
print("\nStep7: Training Stacking Ensemble Model...")
estimators = [
    ('rf',best_rf),
    ('xgb',best_xgb),
    ('ridge',best_ridge),
    ('lasso',best_lasso),
    ('mlp', best_mlp),
    ('knn',best_knn),
]
# 元学习器用Ridge防过拟合
stacking=StackingRegressor(estimators=estimators,final_estimator=Ridge(alpha=1.0),cv=5,n_jobs=-1,passthrough=True)
# 交叉验证
scores=cross_val_score(stacking,X,y,cv=5,scoring=make_scorer(mean_absolute_error),n_jobs=-1)
print(f"Stacking CV MAE:{scores.mean():.4f}±{scores.std():.4f}")

stacking.fit(X,y)

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
y_pred=stacking.predict(X_future)
plt.rcParams['font.sans-serif']=['PingFang SC']
plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(2,1,figsize=(14,10))

# 图1: 历史拟合(考虑圣诞假的误差，取非12月数据的最后14天展示)
y_train_hat=stacking.predict(X)
sub_df=df[df['month']!=12].iloc[-336:]
ts_history=pd.to_datetime(sub_df['timestamp'])

axes[0].plot(ts_history, y.loc[sub_df.index], label='真实值', c='tab:blue', alpha=0.8)
axes[0].plot(ts_history, y_train_hat[sub_df.index], label='拟合值', c='tab:orange', ls='--')
axes[0].set_title('历史数据拟合效果')
axes[0].set_ylabel('能耗(kWh)')
axes[0].legend()
axes[0].grid(True,ls='--',alpha=0.5)

# 图2: 未来72h能耗预测
axes[1].plot(future_df['timestamp'],y_pred,label='预测值',c='tab:red',marker='o',ms=3)
axes[1].set_title('未来一周能耗预测')
axes[1].set_ylabel('预测能耗(kWh)')
axes[1].legend()
axes[1].grid(True,ls='--',alpha=0.5)

plt.tight_layout()
plt.show()

# 保存预测结果
res=pd.DataFrame({'timestamp':future_idx,'predicted_power':y_pred})
res.to_csv('future_week_prediction.csv',index=False)