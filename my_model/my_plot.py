from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .cluster_metric import silhouette_coefficients, silhouette_score, cluster_grouping
from .k_mean import KMeans, centroid as k_mean_centroid
from .k_median import KMedians, centroid as k_median_centroid

def plot_elbow_method(customers, k_max, model='kmean'):
    KFamilyModel = get_model(model)
    sse_list = []
    for i in range(2, k_max + 1):
        km = KFamilyModel(k=i)
        hist = km.fit(customers)
        sse_list.append(hist['SSE'][-1])

    plt.title('Elbow Method ' + KFamilyModel.__name__)
    plt.plot(range(2, k_max + 1), sse_list)
    plt.ylabel('Sum Square Error')
    plt.xlabel('K')
    plt.xticks(range(2, k_max + 1))

    # plt.savefig('fig/elbow_method.png')
    plt.show()

def plot_silhouette_method(customers, k_max, model='kmean'):
    KFamilyModel = get_model(model)
    silhouette_score_list = []
    for i in range(2, k_max + 1):
        km = KFamilyModel(k=i)
        km.fit(customers)
        silhouette_score_list.append(silhouette_score(customers, km.predict(customers)))

    plt.title('Silhouette Method ' + KFamilyModel.__name__)
    plt.plot(range(2, k_max + 1), silhouette_score_list)
    plt.ylabel('Silhouette Score')
    plt.xlabel('K')
    plt.xticks(range(2, k_max + 1))

    # plt.savefig('fig/silhouette_method.png')
    plt.show()

def plot_silhouette_single_k(customers, labels):
    k = len(set(labels))
    plt.xlim([-0.1, 1])
    plt.ylim([0, len(customers) + (k + 1) * 10])

    coefficient_list = silhouette_coefficients(customers, labels)
    avg_score = np.mean(coefficient_list)
    plt.axvline(x=avg_score, color='red', linestyle='--')

    y_lower = 10
    for i in range(k):
        cluster_i_coefficients = coefficient_list[labels == i]
        cluster_i_coefficients.sort()
        cluster_i_size = len(cluster_i_coefficients)

        y_upper = y_lower + cluster_i_size
        color = cm.nipy_spectral(float(i) / k)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_i_coefficients,
            facecolor = color,
            edgecolor = color,
            alpha = 0.7
        )

        plt.text(-0.05, y_lower + 0.5 * cluster_i_size, str(i))
        y_lower = y_upper + 10
    
    plt.title("The silhouette plot for %d clusters" % k)
    plt.xlabel("silhouette coefficient value")
    plt.ylabel("cluster label")
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([])
    plt.show()

def plot_customer_clusters(customers, labels, field_names, model='kmean', outliers=None, title=""):
    centroid_func = get_centroid_func(model)

    k = len(set(labels))
    cluster_list = cluster_grouping(customers, labels)
    centroid_list = np.array([centroid_func(cluster_list[i]) for i in range(k)])

    colors = cm.nipy_spectral(labels.astype(float) / k)
    for idx1 in range(len(customers[0])):
        for idx2 in range(idx1 + 1, len(customers[0])):
            plt.scatter(
                customers[:, idx1], customers[:, idx2], marker=".", s=50, lw=0, alpha=0.7, c=colors, edgecolor="k"
            )

            # Draw cluster centers
            plt.scatter(
                centroid_list[:, idx1],
                centroid_list[:, idx2],
                marker="o",
                c="white",
                alpha=1,
                s=200,
                edgecolor="k",
            )

            for i, c in enumerate(centroid_list):
                plt.scatter(c[idx1], c[idx2], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
            
            # Draw outliers
            if outliers is not None:
                plt.scatter(*zip(*outliers[:, [idx1, idx2]]), marker='o', facecolor='None', edgecolor='r', s=60)

            plt.title("Customer Clusters " + title)
            plt.xlabel(field_names[idx1])
            plt.ylabel(field_names[idx2])
            plt.show()

def plot_train_history(SSE_hist):
    plt.plot(SSE_hist)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title("Training Sum Square Error History")
    plt.xlabel("Iteration")
    plt.ylabel("SSE")
    plt.show()

def get_model(model):
    if model == 'kmedian':
        return KMedians
    else:
        return KMeans

def get_centroid_func(model):
    if model == 'kmedian':
        return k_median_centroid
    else:
        return k_mean_centroid
