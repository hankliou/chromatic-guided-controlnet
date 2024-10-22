from tutorial_dataset import MyDataset

dataset = MyDataset()
print(len(dataset))

item = dataset[1234]
jpg = item['jpg']
txt = item['color_block']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)
