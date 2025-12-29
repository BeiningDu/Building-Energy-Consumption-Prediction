import pandas as pd

file_path = './building_energy.csv'
output_clean_path = 'building_energy_cleaned.csv'

df = pd.read_csv(
    file_path,
    skiprows=1,
    header=None,
    names=['timestamp','Power','airTemperature','dewTemperature','windSpeed','hour','day_of_week','month']
)

df = df.dropna() # 删除缺失值
power_lower_quantile = df['Power'].quantile(0.005)  # 删除极低异常值（0.5%分位数）
df = df[df['Power']>=power_lower_quantile]

df.to_csv(output_clean_path, index=False)