# Mall Customers Clustering (Using K-Means)

## Setup:
- Install libraries in requirements.txt:  
pip install -r requirements.txt

## Run:
Run the following python files in the same order:

  ### choosing_k_analyse.py
  Analyse on choosing the optimized hyperparameter k (number of cluster) by combining 2 methods:
  + Elbow Method
  + Silhouette Method

  ### detect_outlier(before_training).py
  Identify potential outliers by using IQR method on each features of the original dataset

  ### detect_outlier(after_training).py
  Identify potential outliers (after training with k chosen in choosing_k_analyse.py) by using IQR method on the distances between each example and its cluster's centroid
  <br> The potential outliers is circled in the diagram

  ### final_train.py
  Final training using K-means

  ### kmean_kmedian_comparison.py
  Comparison between K-means and K-medians
