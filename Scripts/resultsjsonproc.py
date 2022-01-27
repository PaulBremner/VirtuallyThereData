import pandas as pd
from decimal import Decimal
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pingouin import ancova

df = pd.read_json("results.json", lines=True)

#condition .1 is sound .2 is vision
#even PID is the obvious sound condition
#0.1 is the tutorial scene and can be discarded

print(df.head())

tutorialfilter = df['condition'] > 0.1
df = df[tutorialfilter]

print(df.head())

#replace numbers in condition with labels for sound and vision

df.loc[(df['condition'] == 1.1) | (df['condition'] == 3.1), 'condition'] = 'sound'
df.loc[(df['condition'] == 2.2) | (df['condition'] == 4.2), 'condition'] = 'visual'
print(df.head(10))

#write df to file
df.to_json(path_or_buf="perf_measures.json")

#filter for even PIDs to get the obv condition frame

# obviousfilter = df['PID']%2!=0
# df = df[obviousfilter]

soundDesFilter = df['PID']%2==0
df = df[soundDesFilter]

print(df.head())


print(df.size)

#plt.hist(df['taskTime'])
#plt.show()

#plt.hist(df['radPenalty'])
#plt.show()

#plt.hist(df['collisions'])
#plt.show()

#df = df[(np.abs(stats.zscore(df['taskTime'])) < 3)]
#df = df[df.duplicated(subset=["PID"], keep=False)]
#df = df[(np.abs(stats.zscore(df['radPenalty'])) < 3)]
#df = df[df.duplicated(subset=["PID"], keep=False)]
#df = df[(np.abs(stats.zscore(df['collisions'])) < 3)]
#df = df[df.duplicated(subset=["PID"], keep=False)]
#print(df.size)
proc_cols = ['taskTime', 'radPenalty', 'collisionPenalty']
for c in proc_cols:
    print(c)
    b = df.query('condition == "sound"')[c]
    #plt.hist(b)
    #plt.show()
    #d = stats.boxcox(b)
    #print(b)
    #plt.hist(d)
    #plt.show()
    # e = stats.yeojohnson(b)
    # plt.hist(e)
    # plt.show()
    # stat, p = stats.shapiro(d[0])
    # print(c + " p_boxcox = " + str(p))
    # stat, p = stats.shapiro(e[0])
    # print(c + " p_yeoj = " + str(p))
    # stat, p = stats.shapiro(b)
    # print(c + " p sound = " + str(p))

    a = df.query('condition == "visual"')[c]
    #plt.hist(a)
    #plt.show()
    #f = stats.boxcox(a)
    #plt.hist(f)
    #plt.show()
    #print(a)
    #print(b)
    #print([np.var(x, ddof=1) for x in [a, b]])
    #stat, p = stats.levene(a,b)
    #print(c + " p levene = " + str(p))
    print(a.describe()['mean'])
    print(a.describe()['std'])
    print(b.describe()['mean'])
    print(b.describe()['std'])

    #print(c)
    stat, p = stats.shapiro(a)
    #print(c + " p vis = " + str(p))#put this back in to see the normality test results

    if(p<0.05):
        print(stats.wilcoxon(a, b))
    else:
        print(stats.ttest_rel(a, b))
    #print("normalised data")
    #print(stats.ttest_rel(d[0], f[0]))
    #sns.violinplot(x='condition', y=c, data=df)#, inner="points")
    #plt.show()

# b = df.query('condition == "sound"')['taskTime']
# print(b.describe())
# a = df.query('condition == "visual"')['taskTime']
# print(a.describe())
# print(a.size)
# print("Task Time")
# #print(stats.ttest_rel(a, b))
# print(stats.wilcoxon(a, b))
#
#
# #sns.boxplot(x='condition', y='taskTime', data=df)
# sns.violinplot(x='condition', y='taskTime', data=df)#, inner="points")
# plt.show()
#
# b = df.query('condition == "sound"')['radPenalty']
# print(b.describe())
# a = df.query('condition == "visual"')['radPenalty']
# print(a.describe())
# print("radPenalty")
# #print(stats.ttest_rel(a, b))
# print(stats.wilcoxon(a, b))
#
# #sns.boxplot(x='condition', y='radPenalty', data=df)
# sns.violinplot(x='condition', y='radPenalty', data=df)#, inner="points")
# plt.show()
#
# b = df.query('condition == "sound"')['collisions']
# print(b.describe())
# a = df.query('condition == "visual"')['collisions']
# print(a.describe())
# print("collisions")
# #print(stats.ttest_rel(a, b))
# print(stats.wilcoxon(a, b))
#
# #sns.boxplot(x='condition', y='collisions', data=df)
# sns.violinplot(x='condition', y='collisions', data=df)#, inner="points")
# plt.show()
#
# b = df.query('condition == "sound"')['collisionPenalty']
# print(b.describe())
# a = df.query('condition == "visual"')['collisionPenalty']
# print(a.describe())
# print("collisionPenalty")
# #print(stats.ttest_rel(a, b))
# print(stats.wilcoxon(a, b))
#
# #sns.boxplot(x='condition', y='collisions', data=df)
# sns.violinplot(x='condition', y='collisionPenalty', data=df)#, inner="points")
# plt.show()