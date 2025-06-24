file = open('loss.txt', 'r', encoding='utf-8')

steps = [ 0.0 ]

for line in file:
    values = line.strip().split(',')
    
    if len(values) == 3:
        step = int(values[0])   # 假設 step 是整數
        epoch = int(values[1])  # 假設 epoch 是整數
        loss = float(values[2]) # 假設 loss 是小數

        if step < steps[-1]:
            print(step)
        
        steps.append(step)