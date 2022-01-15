import matplotlib.pyplot as plt


x = [1,2,3,4,5, 7,8,9,10,11, 13,14,15,16,17, 19,20,21,22,23, 30]

height = [1.669656500379998, 1.6823152823577907, 1.7386721165241137, 1.7983275662742169, 1.8517293529085943, 
          1.670008018600577, 1.6941286293023496, 1.7748166026896126, 1.8496148092925597, 1.9320969043049259,
          1.696563393639221, 1.7269350509454864, 1.882169442047094, 2.0194891845501854, 2.1366734642286684,
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

plt.savefig("plots/clf_plot.png")