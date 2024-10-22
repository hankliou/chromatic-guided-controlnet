import matplotlib.pyplot as plt

def read_data(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            party, value = line.strip().split()
            data[party] = float(value)
    return data

def plot_data(data):
    parties = list(data.keys())
    values = list(data.values())
    
    plt.ylim(27, 29)
    plt.bar(parties, values)
    plt.xlabel('files')
    plt.ylabel('psnr')
    plt.title('psnr for every pic in inference')
    plt.xticks(rotation=45)
    plt.savefig('psnr.png')
    plt.show()

def main():
    file_path = 'psnr.txt'  # 修改為你的檔案路徑
    data = read_data(file_path)
    plot_data(data)

if __name__ == "__main__":
    main()
