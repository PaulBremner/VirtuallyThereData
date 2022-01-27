import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

columns = list(range(17,90))
df = pd.read_csv("Arm_TeleOp_numcoded.csv", usecols=columns, skiprows=[1,2])
#print(df.head(10))
#print(df.dtypes)

# obviousfilter = df['Q1']%2!=0
# df = df[obviousfilter]
# df.reset_index(inplace=True)
# print(df.head(10))

soundDesFilter = df['Q1']%2==0
df = df[soundDesFilter]
df.reset_index(inplace=True)
print(df.head(10))

age = df['Q35'].describe()
print(age)
print(df['Q34'].value_counts())

df2 = pd.read_json("perf_measures.json")
#

# obviousfilter = df2['PID']%2!=0
# df2 = df2[obviousfilter]
# df2.reset_index(inplace=True)
# print(df2.head(10))

soundDesFilter = df2['PID']%2==0
df2 = df2[soundDesFilter]
df2.reset_index(inplace=True)
print(df2.head(10))

#need to filter all sound responses and all visual responses using df2 data - each PID has two entries which shows the order of conditions
soundlist = []

#print(df.iloc[:,np.r_[0:3,3:38]])

IPQ = [] #23, 27
for i in range(1,10):
    IPQ.append("IPQ" + str(i))

SUS = []  #22, 28
for i in range(1,11):
    SUS.append("SUS" + str(i))

NTLX = []#24. 29
for i in range(1,7):
    NTLX.append("NTLX" + str(i))

NTLXW = [] #26, 30
for i in range(1,7):
    NTLXW.append("NTLXW" + str(i))

Stress = [] #25, 31
for i in range(1,5):
    Stress.append("Stress" + str(i))

cols = IPQ + SUS + NTLX + NTLXW + Stress

df_questions = pd.DataFrame(columns=cols)

# create 1 df with PID, Sex, Age, + IPQ + SUS + NTLX + NTLXW + Stress + condition
# so each PID should have 2 rows 1 for each condition
# hence each row will have the PID and condition columns from df2, then sex and age,
# then the correct set of columns - the first set for the first row for that pid and the 2nd for the second

dfSlice1 = df.iloc[:,np.r_[4:39]]
dfSlice1.columns = df_questions.columns
dfSlice2 = df.iloc[:,np.r_[39:74]]
dfSlice2.columns = df_questions.columns

#labels = ['PID', 'condition']
#df_processed = pd.DataFrame(columns=labels)

#create a df slice from df2 that just has the PID and condition cols
#create a df by sequentially adding rows from each of the 2 df slices
#stick the 2 dfs together

for i in range(len(df)):
    df_questions = df_questions.append(dfSlice1.loc[i], ignore_index=True)
    df_questions = df_questions.append(dfSlice2.loc[i], ignore_index=True)

print(df_questions.head())
print("df2 sz " + str(len(df2)))
print("dfq sz " + str(len(df_questions)))

df_final = df2.filter(['PID', 'condition'], axis=1)#.append(df_questions)
df_final = pd.concat([df_final, df_questions], axis=1)
df_final['IPQ3'] = 6 - df_final['IPQ3']
df_final['IPQ6'] = 6 - df_final['IPQ6']
df_final['NTLX4'] = 10 - df_final['NTLX4']

for i in [2,4,5,8,10]:
    df_final['SUS' + str(i)] = 6 - df_final['SUS' + str(i)]

for c in NTLXW:
    df_final[c] = 6 - df_final[c]

df_final['P'] =df_final['IPQ1']
df_final['SP'] =df_final[['IPQ2','IPQ3','IPQ4']].mean(axis=1)
df_final['INV'] =df_final[['IPQ5','IPQ6']].mean(axis=1)
df_final['EXP'] =df_final[['IPQ7','IPQ8','IPQ9']].mean(axis=1)

df_final['SUS'] = df_final[SUS].sum(axis=1)
df_final['SUS'] = (df_final['SUS'] -10)*2.5

for c in NTLX:
    df_final[c] = df_final[c] * df_final[c[:4] + 'W' + c[4:]]

df_final['NTLX'] = df_final[NTLX].mean(axis=1)

#pd.set_option('display.max_columns', None)
print(df_final.head())

proc_cols = Stress + ['P', 'SP', 'INV', 'EXP', 'SUS', 'NTLX']

for c in proc_cols:
    stat, p = stats.shapiro(df_final[c])
    print(c + " p = " + str(p))

# stress and presence data not normal so use non-parametric tests
df_final[''] = ""

for c in proc_cols:
    print(c)
    b = df_final.query('condition == "sound"')[c]
    a = df_final.query('condition == "visual"')[c]
    a = a.dropna()
    b = b.dropna()

    #print(a)
    print(b.describe())
    print(a.describe())
    #sns.violinplot(x = '', hue='condition', y=c, data=df_final)#, split=True)#, inner="points")
    #plt.show()

    if c != 'SUS' and c != 'NTLX':
        print(stats.wilcoxon(a, b))
    else:
        print(stats.ttest_rel(a, b))


# plt.hist(df_final['P'])
# plt.show()
#
# plt.hist(df_final['SP'])
# plt.show()
#
# plt.hist(df_final['INV'])
# plt.show()
#
# plt.hist(df_final['EXP'])#
# plt.show()
#
# plt.hist(df_final['SUS'])#
# plt.show()
#
# plt.hist(df_final['NTLX'])
# plt.show()

#TODO add cols for the processed data for each questionnaire
# IPQ
# Reverse coding:
# 23_3
# 23_6
# 1 - pres
# 2,3,4 - Spatial presence
# 5,6 - Involvement
# 7,8 - Experienced Realism
#
# NTLX
# 4 is reverse scored
# Mean of all ratings*weight for that rating (0-5) = workload score. Low score means low workload
#
# SUS
# Reverse coded:
# 2,4,5, 8, 10
# After reverse coding sum all answers - 10 and *2.5. >68 is good
#
# For stress do each item individually