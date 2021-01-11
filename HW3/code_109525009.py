import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mf_sgd(R, K=64, alpha=1e-4, beta=1e-2, iterations=10):
    """
    :param R: user-item rating matrix
    :param K: number of latent dimensions
    :param alpha: learning rate
    :param beta: regularization parameter
    """
    num_users, num_items = R.shape

    # Initialize user and item latent feature matrice
    P = np.random.normal(scale=1. / K, size=(num_users, K))
    Q = np.random.normal(scale=1. / K, size=(num_items, K))

    # Initialize the biases
    b_u = np.zeros(num_users)
    b_i = np.zeros(num_items)
    b = np.mean(R[np.where(R != 0)])

    # Create a list of training samples
    #i是user, j是movie, R[i, j]是分數
    #samples 是原有真實分數
    samples = [
        (i, j, R[i, j])
        for i in range(num_users)
        for j in range(num_items)
        if R[i, j] > 0
    ]

    # Perform stochastic gradient descent for number of iterations
    training_loss = []
    for iters in range(iterations):
        np.random.shuffle(samples)

        for i, j, r in samples:
          pred_r = np.dot( P[i], Q[j] ) + b_u[i] + b_i[j] + b
          d = r - pred_r
          b_u[i] = b_u[i] - alpha*( -d + beta*b_u[i] )
          b_i[j] = b_i[j] - alpha*( -d + beta*b_i[j] )
          P[i] = P[i] - alpha*( -d*Q[j] + beta*P[i] )
          Q[j] = Q[j] - alpha*( -d*P[i] + beta*Q[j] )

        # Using RMSE to compute training_loss
        xs, ys = R.nonzero()
        pred = b + b_u[:, np.newaxis] + b_i[np.newaxis:, ] + P.dot(Q.T)
        error = 0
        for x, y in zip(xs, ys):
            error += pow(R[x, y] - pred[x, y], 2)
        rmse = np.sqrt(error / len(xs))
        training_loss.append((iters, rmse))
        print("Iteration: %d ; error = %.4f" % (iters + 1, rmse))

        # if (iters + 1) % 10 == 0:
        #     print("Iteration: %d ; error = %.4f" % (iters + 1, rmse))

    return pred, b, b_u, b_i, training_loss


def plot_training_loss(training_loss):
    x = [x for x, y in training_loss]
    y = [y for x, y in training_loss]
    plt.figure(figsize=(16, 4))
    plt.plot(x, y)
    plt.xticks(x, x)
    plt.xlabel("Iterations")
    plt.ylabel("Root Mean Square Error")
    plt.grid(axis="y")
    plt.savefig("training_loss.png")
    plt.show()



if __name__ == "__main__":
    data = pd.read_csv('ratings.csv')
    table = pd.pivot_table(data, values='rating', index='userId', columns='movieId', fill_value=0)
    R = table.values
    pred, b, b_u, b_i, loss = mf_sgd(R)
    print("P x Q:")
    print(pred)
    print("Global bias:")
    print(b)
    print("User bias:")
    print(b_u)
    print("Item bias:")
    print(b_i)
    plot_training_loss(loss)
