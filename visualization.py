import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

file_path = './building_energy.csv'
df = pd.read_csv(
    file_path,
    skiprows=1,
    header=None,
    names=['timestamp','Power','airTemperature','dewTemperature','windSpeed','hour','day_of_week','month'],
    parse_dates=['timestamp'],
    date_parser=lambda x:pd.to_datetime(x.strip(),format='%Y-%m-%d %H:%M:%S'),
) # 读取数据，根据列特征分类
df.set_index('timestamp',inplace=True) # 时间戳为索引

print("数据形状:",df.shape)
print(df.head())
print(df.isnull().sum())
df = df[df['Power'].notna()]
df = df[df['Power']>=0] # 数据预览和缺失值清洗

df_daily = df['Power'].resample('D').mean()
plt.figure(figsize=(14,5))
plt.plot(df_daily.index,df_daily.values,color='steelblue')
plt.title('建筑日均能耗时间序列')
plt.xlabel('日期')
plt.ylabel('日均能耗(kWh)')
plt.grid(True,linestyle='--',alpha=0.6)
plt.tight_layout()
plt.show() # 日均能耗趋势图

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['airTemperature'],y=df['Power'],alpha=0.4)
plt.title('能耗与环境温度关系')
plt.xlabel('环境温度(°C)')
plt.ylabel('能耗(kWh)')
plt.grid(True,linestyle='--',alpha=0.6)
plt.tight_layout()
plt.show() # 能耗与环境温度的散点图

hourly_stats = df.groupby('hour')['Power'].agg(['mean','std']).reset_index()
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(hourly_stats['hour'],hourly_stats['mean'],marker='o',color='tab:red')
plt.title('逐时平均能耗')
plt.xlabel('小时 (0–23)')
plt.ylabel('平均能耗 (kWh)')
plt.grid(True,linestyle='--',alpha=0.5)
plt.subplot(1,2,2)
plt.bar(hourly_stats['hour'],hourly_stats['std'],color='tab:blue')
plt.title('逐时能耗标准差')
plt.xlabel('小时 (0–23)')
plt.ylabel('标准差')
plt.grid(True,linestyle='--',alpha=0.5)
plt.tight_layout()
plt.show() # 逐时能耗平均值和标准差

dow_labels = ['Mon','Tues','Wedn','Thur','Fri','Sat','Sun']
dow_stats = df.groupby('day_of_week')['Power'].agg(['mean','std']).reindex(range(7)).reset_index()
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(dow_stats['day_of_week'],dow_stats['mean'],marker='o',color='tab:red')
plt.title('每周各天平均能耗')
plt.xlabel('星期')
plt.xticks(ticks=range(7),labels=dow_labels)
plt.ylabel('平均能耗 (kWh)')
plt.grid(True,linestyle='--',alpha=0.5)
plt.subplot(1,2,2)
plt.bar(dow_stats['day_of_week'],dow_stats['std'],color='tab:blue')
plt.title('每周各天能耗标准差')
plt.xlabel('星期')
plt.xticks(ticks=range(7),labels=dow_labels)
plt.ylabel('标准差')
plt.grid(True,linestyle='--',alpha=0.5)
plt.tight_layout()
plt.show() # 逐日（每周的第几天）能耗平均值和标准差

month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthly_stats = df.groupby('month')['Power'].agg(['mean','std']).reindex(range(1, 13)).reset_index()
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(monthly_stats['month'],monthly_stats['mean'],marker='o',color='tab:red')
plt.title('每月平均能耗')
plt.xlabel('月份')
plt.xticks(ticks=range(1,13),labels=month_labels)
plt.ylabel('平均能耗 (kWh)')
plt.grid(True,linestyle='--',alpha=0.5)
plt.subplot(1,2,2)
plt.bar(monthly_stats['month'],monthly_stats['std'],color='tab:blue')
plt.title('每月能耗标准差')
plt.xlabel('月份')
plt.xticks(ticks=range(1,13),labels=month_labels)
plt.ylabel('标准差')
plt.grid(True,linestyle='--',alpha=0.5)
plt.tight_layout()
plt.show() # 逐月能耗平均值和标准差