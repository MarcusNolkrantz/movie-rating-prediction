import matplotlib.pyplot as plt


x = [1,2,3,4,5, 7,8,9,10,11, 13,14,15,16,17, 19,20,21,22,23, 30]

height = [2.5852225886232483, 2.5985160758450125, 2.695795548227535, 2.7948268755152514, 2.815539983511954, 
          2.5882110469909314, 2.6155193734542457, 2.707028029678483, 2.8388293487221765, 2.833779884583677,
          2.647361912613355, 2.7084707337180545, 2.81760098928277, 2.8441879637262986, 2.8490313272877166,
          8.399422918384172, 8.399422918384172, 8.399422918384172, 8.399422918384172, 8.399422918384172, 
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

plt.savefig("plots/reg_plot.png")