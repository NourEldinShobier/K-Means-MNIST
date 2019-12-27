from utils.fileManager import FileManager
from KMeans import KMeans
from utils.fns import plot_images


def main():
    # 1) load data
    train_set, train_labels = FileManager.load_data_set()

    plot_images(train_set)

    # 2) fit data
    k_means = KMeans(train_set, train_labels, k=15, max_iterations=100, eps=1e-10)
    k_means.fit()


main()
