import matplotlib.pyplot as plt

file = open('loss.txt', 'r', encoding='utf-8')

data = []
minloss_epoch = [999, 0]

for line in file:
    values = line.strip().split(',')
    
    if len(values) == 3:
        step = int(values[0])   # 假設 step 是整數
        epoch = int(values[1])  # 假設 epoch 是整數
        loss = float(values[2]) # 假設 loss 是小數
        
        # 刪除重疊的資料
        while data and step < data[-1]['step']:
            data.pop()

        # 將 step, epoch, loss 存入字典
        data_dict = {
            'step': step,
            'epoch': epoch,
            'loss': loss
        }
        
        data.append(data_dict)

epoch_loss = []
for i in range(1000):
    epoch_loss.append({'sum': 0, 'cnt': 0})
    
for d in data:
    epoch_loss[d['epoch']]['sum'] += d['loss']
    epoch_loss[d['epoch']]['cnt'] += 1

cnt = 0
for e in epoch_loss:
    if e['cnt'] != 0:
        e['sum'] /= e['cnt']
        print(f'{cnt}: {e["sum"]}')
        
        if e['sum'] < minloss_epoch[0]:
            minloss_epoch = [e['sum'], cnt]
            
        cnt+=1
    
print(minloss_epoch)

plt.plot([item['sum'] for item in epoch_loss if item['sum'] != 0], linestyle='-', color='b')
plt.savefig('step_vs_loss.png')