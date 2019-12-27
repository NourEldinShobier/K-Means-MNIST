import numpy as np
import matplotlib.pyplot as plt
import random
import copy

from tqdm import tqdm

from utils.fns import plot_centroids, plot_images


class KMeans:

    def __init__(self, train_set, train_labels, k=10, max_iterations=200, eps=1e-10):
        self.train_set = train_set
        self.train_labels = train_labels

        self.k = k
        self.eps = eps
        self.accuracy = 0
        self.iterations = 0
        self.max_iterations = max_iterations

        # Loss of each iteration.
        self.losses = []
        self.centroids = []

        self.clusters = {}
        self.clusters_labels = []
        self.clusters_info = []
        self.clusters_accuracy = []

        # initialize centroid with 0
        self.predicted_labels = [None] * 60000
        self.current_centroids = [[0.0] * (28 * 28)] * self.k

    def fit(self):
        # 1) fit data
        self.start()

        # 2) plot centroids
        plot_centroids(self.centroids)

        # 3) plot loss
        plt.plot(range(self.iterations), self.losses)
        plt.show()

        # 4) display clusters
        for key, data in list(self.clusters['data'].items()):
            print('Cluster:', key, 'Label:', self.clusters_labels[key])
            plot_images(data[:min(25, data.shape[0])])

        # 5) calculate accuracy
        print('Accuracy:', self.accuracy)

    def start(self):
        # Randomly initialize k means (or centroids).
        self.init_centroids()

        # Repeat until convergence.
        while not self.check_convergence(self.current_centroids, self.centroids):
            self.current_centroids = copy.deepcopy(self.centroids)
            self.init_clusters()
            for image_index, image in tqdm(enumerate(self.train_set)):
                # Initialize distance
                min_distance = float('inf')
                for centroid_index, centroid in enumerate(self.centroids):
                    # Get the nearest cluster for this image.
                    distance = np.linalg.norm(image - centroid)
                    if distance < min_distance:
                        min_distance = distance
                        self.predicted_labels[image_index] = centroid_index

                # Assign each image to the closest centroid.
                if self.predicted_labels[image_index] is not None:
                    self.clusters['data'][self.predicted_labels[image_index]].append(image)
                    self.clusters['labels'][self.predicted_labels[image_index]].append(self.train_labels[image_index])

            self.reshape_cluster()
            self.update_centroids()
            self.calculate_loss()
            self.iterations += 1
        self.calculate_accuracy()

    def init_centroids(self):
        for i in range(self.k):
            random_index = random.choice(range(len(self.train_set)))
            self.centroids.append(self.train_set[random_index])

    def init_clusters(self):
        self.clusters = {
            'data': {i: [] for i in range(self.k)},
            'labels': {i: [] for i in range(self.k)}
        }

    def update_centroids(self):
        for i in range(self.k):
            cluster = self.clusters['data'][i]
            if len(cluster) == 0:
                random_index = random.choice(range(len(self.train_set)))
                self.centroids[i] = self.train_set[random_index]
            else:
                self.centroids[i] = np.mean(np.vstack((self.centroids[i], cluster)), axis=0)

    def reshape_cluster(self):
        for ID, mat in list(self.clusters['data'].items()):
            self.clusters['data'][ID] = np.array(mat)

    def check_convergence(self, old_centroids, new_centroids):
        if self.iterations > self.max_iterations:
            return True

        distance = np.linalg.norm(np.array(new_centroids) - np.array(old_centroids))

        if distance <= self.eps:
            print("Converged, Distance:", distance)
            return True

        print(f"\nIteration: {self.iterations}, Distance: {distance}")
        return False

    def calculate_loss(self):
        loss = 0
        for key, value in list(self.clusters['data'].items()):
            if value is not None:
                for v in value:
                    loss += np.linalg.norm(v - self.centroids[key])
        self.losses.append(loss)

    def calculate_accuracy(self):
        for cluster, labels in list(self.clusters['labels'].items()):
            if isinstance(labels[0], np.ndarray):
                labels = [l[0] for l in labels]

            occur = 0
            max_label = max(set(labels), key=labels.count)
            self.clusters_labels.append(max_label)
            for label in labels:
                if label == max_label:
                    occur += 1
            acc = occur / len(list(labels))

            self.clusters_info.append([max_label, occur, len(list(labels)), acc])
            self.clusters_accuracy.append(acc)
            self.accuracy = sum(self.clusters_accuracy) / self.k
