import pandas as pd
USER_NUM = "01"
def read_csv(path: str):
    epsar = pd.read_csv(path, names=("EP", "S2", "S1","A","R"))
    return epsar

fname='user'+USER_NUM+'/user_'+USER_NUM+'_報酬設定B_状態行動報酬の組み合わせ.csv'

epsar = read_csv(fname)

# 各状態に対する与える報酬の確率

C = 6
Pr = []
for i in range(C):
    x = epsar.query('S1 == @i')
    lx = len(x)
    r1 = x.query('R == 1')
    rm1 = x.query('R == -1')
    r0 = x.query('R == 0')
    Pr.append([lx, len(r1)/lx, len(r0)/lx, len(rm1)/lx])
# print(Pr)

Pr1 = []
for i in range(C):
    x = epsar.query('A == @i & S1 == @i')
    lx = len(x)
    r1 = x.query('R == 1')
    rm1 = x.query('R == -1')
    r0 = x.query('R == 0')
    Pr1.append([lx, len(r1)/lx, len(r0)/lx, len(rm1)/lx])
Pr2 = []
for i in range(C):
    x = epsar.query('S2 == @i & S1 == @i & A == @i')
    if len(x) == 0:
        Pr2.append([0,0,0,0])
        continue
    lx = len(x)
    r1 = x.query('R == 1')
    rm1 = x.query('R == -1')
    r0 = x.query('R == 0')
    Pr2.append([lx, len(r1)/lx, len(r0)/lx, len(rm1)/lx])

import csv
brank = [[]]
with open('user'+USER_NUM+'/out.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(Pr)
    writer.writerows(brank)    
    writer.writerows(Pr1)
    writer.writerows(brank)    
    writer.writerows(Pr2)

epsar[['S1', 'A', 'R']].to_csv('user'+USER_NUM+'/sar.csv')
