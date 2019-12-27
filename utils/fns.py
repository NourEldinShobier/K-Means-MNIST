import numpy as np
import matplotlib.pyplot as plt


def plot_images(images):
    rows = 4
    columns = 4

    fig, ax = plt.subplots(rows, columns, figsize=(rows, columns))

    for i in range(rows):
        for j in range(columns):
            random_index = np.random.choice(range(len(images)))
            picked_image = images[random_index].copy()

            picked_image = picked_image.reshape(1, 28, 28).astype('uint8')
            ax[i][j].set_axis_off()
            ax[i][j].imshow(picked_image[0])

    plt.show()


def plot_centroids(centroids):
    columns = len(centroids)

    fig, ax = plt.subplots(columns)

    index = 0
    for i in range(columns):
        picked_image = centroids[index].copy()

        picked_image = picked_image.reshape(1, 28, 28).astype('uint8')
        ax[i].set_axis_off()
        ax[i].imshow(picked_image[0])
        index += 1

    plt.show()
