import matplotlib.pyplot as plt

from .cluster_metric import silhouette_score
from .k_mean import KMeans

def plot_elbow_method(customers, k_max):
    final_sse_list = []
    for i in range(2, k_max + 1):
        km = KMeans(k=i)
        hist = km.fit(customers)
        final_sse_list.append(hist['SSE'][-1])

    plt.plot(range(2, k_max + 1), final_sse_list)
    plt.title('Elbow Method')
    plt.ylabel('Sum Square Error')
    plt.xlabel('K')
    plt.xticks(range(2, k_max + 1))
    # plt.savefig('fig/elbow_method.png')
    plt.show()

def plot_silhouette_method(customers, k_max):
    silhouette_score_list = []
    for i in range(2, k_max + 1):
        km = KMeans(k=i)
        km.fit(customers)
        silhouette_score_list.append(silhouette_score(customers, km.predict(customers)))

    plt.plot(range(2, k_max + 1), silhouette_score_list)
    plt.title('Silhouette Method')
    plt.ylabel('Silhouette Score')
    plt.xlabel('K')
    plt.xticks(range(2, k_max + 1))
    # plt.savefig('fig/silhouette_method.png')
    plt.show()
