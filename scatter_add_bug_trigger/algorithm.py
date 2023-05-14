import time
import torch
import pandas as pd
# torch.cuda.synchronize()
def convert_date(date_str):
    res = date_str.split('-')
    return int(res[0] + res[1] + res[2])

def convert_char(char):
    return ord(char)

def generate_group(group_by: list,device,length):

    start = time.time()
    group_hash = torch.zeros(length, device=device)
    for group_column in group_by:
        out,inverse = torch.unique(group_column,return_inverse = True)
        count = out.shape[0]
        group_hash = group_hash * count + inverse
    end = time.time()
    print("group: calculate group hash:", end - start)

    start = time.time()
    output, inverse, counts = torch.unique(group_hash, return_inverse=True, return_counts=True)
    group_num = output.shape[0]
    end = time.time()
    print("group: unique inverse:", end - start)
    start = time.time()
    select_id = torch.zeros(group_num, dtype=torch.int64, device=device).scatter_(dim=0, index=inverse,
                                                                                  src=torch.arange(length,
                                                                                                   device=device))
    end = time.time()
    print("group: select id:", end - start)
    return  group_num, inverse, counts,select_id

def sum_over_group(data,group_num,group_id,device):
        start = time.time()
        res = torch.zeros(group_num,dtype=data.dtype,device=device)
        res = torch.scatter_add(res,0,group_id,data)
        # print(res)
        end  = time.time()

        print('sum over group:',end-start)

        return res
def avg_over_group(data,group_num,group_id,group_count,device):

        start = time.time()
        res = torch.zeros(group_num, dtype=data.dtype, device=device)
        res = torch.scatter_add(res,0,group_id,data)/group_count
        # print(res)
        end = time.time()
        print('avg over group:', end - start)
        return res

# read csv file
df = pd.read_csv('../../csvs/lineitem.csv')

print(df)
# set device
device = 'cpu'

#preprocessing and transform into tensor
L_QUANTITY = torch.tensor(df['L_QUANTITY'].values,device=device)
L_EXTENDEDPRICE = torch.tensor(df['L_EXTENDEDPRICE'].values,device=device)
L_DISCOUNT = torch.tensor(df['L_DISCOUNT'].values,device=device)
L_TAX = torch.tensor(df['L_TAX'].values,device=device)
L_RETURNFLAG = torch.tensor(df['L_RETURNFLAG'].apply(convert_char).values,device=device)
L_LINESTATUS = torch.tensor(df['L_LINESTATUS'].apply(convert_char).values,device=device)
L_SHIPDATE = torch.tensor(df['L_SHIPDATE'].apply(convert_date).values,device=device)

print('already load table')
# start calculation
start = time.time()

mask = torch.le(L_SHIPDATE,19981201)
L_QUANTITY = L_QUANTITY[mask]
L_DISCOUNT = L_DISCOUNT[mask]
L_EXTENDEDPRICE = L_EXTENDEDPRICE[mask]
L_TAX = L_TAX[mask]
L_RETURNFLAG = L_RETURNFLAG[mask]
L_LINESTATUS = L_LINESTATUS[mask]


temp1 = (1-L_DISCOUNT)*L_EXTENDEDPRICE
temp2 = temp1*(L_TAX+1)
group_num, group_id, counts,select_id = generate_group([L_RETURNFLAG,L_LINESTATUS],device,L_RETURNFLAG.shape[0])

res1 = L_RETURNFLAG[select_id]
res2 = L_LINESTATUS[select_id]

res3 = sum_over_group(L_QUANTITY,group_num, group_id,device)

res4 = sum_over_group(L_EXTENDEDPRICE,group_num, group_id,device)
res5 = sum_over_group(temp1,group_num, group_id,device)
res6 = sum_over_group(temp2,group_num, group_id,device)

res7 = avg_over_group(L_QUANTITY,group_num, group_id,counts,device)
res8 = avg_over_group(L_EXTENDEDPRICE,group_num, group_id,counts,device)
res9 = avg_over_group(L_DISCOUNT,group_num, group_id,counts,device)
res10 = counts
#finish calculation
print('finished')
end = time.time()

res = [res1, res2, res3, res4, res5, res6, res7, res8, res9, res10]
print('====',end-start,'====')

# print the result
start = time.time()
for i in res:
    print(i)
end = time.time()
print('====',end-start,'====')







