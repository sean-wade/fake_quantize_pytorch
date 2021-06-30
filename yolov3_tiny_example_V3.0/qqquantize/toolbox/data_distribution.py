
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

out_dir = './temp/'
os.makedirs(out_dir, exist_ok=True)

if __name__ == '__main__':
    data = pickle.load(open('../hist.pkl', 'rb'))
    hist_range = data['hist_range']
    hist = data['hist']
    for name in hist_range.keys():
        print(name)
        min_val, max_val = hist_range[name]
        h = hist[name].cpu().numpy()
        x = np.arange(min_val, max_val, (max_val - min_val) / len(h))
        plt.bar(x, height=h, width=0.01)
        plt.title(name)
        plt.savefig(out_dir + name + '.png')
        plt.close('all')