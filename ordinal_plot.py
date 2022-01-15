import matplotlib.pyplot as plt


x = [1,2,3,4,5, 7,8,9,10,11, 13,14,15,16,17, 19,20,21,22,23, 30]

height = [2.3933055151226297, 2.4564497806961048, 2.6313323597227947, 2.805344445589448, 2.8866193225731656, 
          2.4020615693270817, 2.5030379708110058, 2.728555073938582, 2.910297742876649, 2.9962549673459398,
          2.449407943889118, 2.5850007889787716, 2.90088937229493, 3.1920468927117684, 3.3770098831667354,
          4.605145107687552, 4.593436823732482, 4.580392106347897, 4.574523070383347, 4.572288167250618, 
          0]
width = 0.5
color = ["r","r","r","r","r", "b","b","b","b","b", "m","m","m","m","m", "g","g","g","g","g"]
tick_label = ["epoch 1", "epoch 2", "epoch 3", "epoch 4", "epoch 5", "epoch 1", "epoch 2", "epoch 3", "epoch 4", "epoch 5",
              "epoch 1", "epoch 2", "epoch 3", "epoch 4", "epoch 5", "epoch 1", "epoch 2", "epoch 3", "epoch 4", "epoch 5", "" ]
plt.bar(x, height, width=width, tick_label=tick_label, color=color)
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.3)

colors = {'lr 8e^-6':'red', 'lr 1e^-5':'b', 'lr 2e^-5':'magenta', 'lr 5e^-5':'green'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
plt.ylabel("Validation loss")

plt.savefig("plots/ord_plot.png")