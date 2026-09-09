import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Datani fayldan oxuyuruq
df = pd.read_csv('medical_examination.csv')

# 2: 'overweight' (artıq çəki) sütununu əlavə edirik (BMI > 25 olduqda 1, əks halda 0)
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3: Normalizasiya: 1 dəyərini 0 (yaxşı), 1-dən böyük dəyərləri isə 1 (pis) edirik
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4
def draw_cat_plot():
    # 5: pd.melt ilə datanı kateqorik formata salırıq
    df_cat = pd.melt(
        df, 
        id_vars=['cardio'], 
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6: Datanı 'cardio', 'variable' və 'value' üzrə qruplaşdırıb saylarını 'total' sütununa yazırıq
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7: Seaborn catplot vasitəsilə sütunlu diaqram qururuq
    catplot = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar'
    )

    # 8: Şəkli saxlayacaq fig obyektini götürürük
    fig = catplot.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11: Yanlış/kənar göstəriciləri temizləyirik (Filterləmə)
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12: Korrelyasiya matrisini hesablayırıq
    corr = df_heat.corr()

    # 13: Matrisin yuxarı üçbucağı üçün maska (mask) yaradırıq
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14: Matplotlib fig və ax obyektlərini yaradırıq
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15: Seaborn heatmap ilə korrelyasiya xəritəsini çəkirik
    sns.heatmap(
        corr,
        annot=True,
        fmt='.1f',
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.5},
        ax=ax
    )

    # 16
    fig.savefig('heatmap.png')
    return fig