from mlp_model import *
from gen_data import *
from matplotlib import pyplot as plt


model = mlp(2, seed=3)
model.add_layer(3, 'none')
model.add_layer(2, 'none')
model.add_layer(1, 'none')
sample = 100
split = 0.9

# training with xor data
x, y = generate_XOR_easy()
sample = x.shape[0]

# training with linear data
# x, y = generate_linear(sample)
# sample = x.shape[0]

train_x, train_y, test_x, test_y = x[:int(sample*split)], y[:int(sample*split)], x[int(sample*split):], y[int(sample*split):]
model.fit(train_x, train_y, epochs=1000, batch_size=1)

plt.plot(model.history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')

print(f'testing data... (with activation function)')
y_pred = model.test(x.copy(), y.copy())
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm')
plt.title('Ground truth')
plt.subplot(1, 2, 2)
plt.scatter(x[:, 0], x[:, 1], c=y_pred, cmap='coolwarm')
plt.title('Prediction')
plt.tight_layout()
plt.show()
