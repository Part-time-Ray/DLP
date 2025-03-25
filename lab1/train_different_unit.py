from mlp_model import *
from gen_data import *
from matplotlib import pyplot as plt

sample = 100
split = 0.9

# training with xor data
x, y = generate_XOR_easy()
sample = x.shape[0]

# training with linear data
# x, y = generate_linear(sample)
# sample = x.shape[0]

train_x, train_y, test_x, test_y = x[:int(sample*split)], y[:int(sample*split)], x[int(sample*split):], y[int(sample*split):]

for ind, unit in enumerate([4, 8, 16, 32]):
    print('Unit: ', unit)
    model = mlp(2)
    model.add_layer(unit, 'relu')
    model.add_layer(unit//2, 'relu')
    model.add_layer(1, 'sigmoid')
    model.fit(train_x.copy(), train_y.copy(), epochs=1000, batch_size=1)

    plt.subplot(2, 2, ind+1)
    plt.plot(model.history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Model Loss Over Epochs ({unit=})')

    print(f'testing data... ({unit=})')
    y_pred = model.test(test_x.copy(), test_y.copy())
    print('-'*50)
plt.tight_layout()
plt.show()