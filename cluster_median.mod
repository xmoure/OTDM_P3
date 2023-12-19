# Num of points
param m; 
set M:={1..m};

# Num of clusters
param k;

# Euclidean distances matrix
param d{M,M};

# Matrix of the results
var x{M,M} binary;

# Cost function
minimize TotalDistance: 
  sum{i in M, j in M} d[i, j] * x[i, j];

# Constraints
# Constraint 1: Every point belongs to one cluster
subject to OneClusterPerPoint {i in M}: 
  sum{j in M} x[i, j] = 1;

# Constraint 2: Exactly k clusters
subject to ExactlyKClusters:
  sum{j in M} x[j, j] = k;

# Constraint 3: A point may belong to a cluster only if the cluster exists
subject to ClusterExists {i in M, j in M}:
  x[j, j] >= x[i, j];
