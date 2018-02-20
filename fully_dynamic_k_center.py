from __future__ import generators
import math
from scipy import spatial
from copy import deepcopy
import h5py
import numpy as np

#########################
# Min and max distances #
#########################

def Haversine(point, neighbor):
    """
    The function takes two tuples (two points) and computes their Haversine distance
    """
    lat1 = point[0]
    lat2 = neighbor[0]
    lon1 = point[1]
    lon2 = neighbor[1]
    R=6371000                             
    phi_1=math.radians(lat1)
    phi_2=math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2.0)**2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
    return R*c/1000


def min_distance(dataset):
    """
    This function computes the distance between the two closest points of the dataset.
    It is based on the Delaunay triangulation (by definition the closest pair of
    points in a set define an edge in the Delaunay triangulation of this set). We 
    adapted a bit the original version (that relies on the Euclidean distance), to
    consider instead the Haversine distance
    """
    # set up the triangulation
    mesh = spatial.Delaunay(dataset)
    edges = np.vstack((mesh.vertices[:,:2], mesh.vertices[:,-2:]))
    points1 = mesh.points[edges[:,0]]
    points2 = mesh.points[edges[:,1]]
    
    # here we adapted the code to use the Haversine distance instead of the Euclidean
    dists = ([Haversine(p1, p2) for p1, p2 in zip(points1, points2)])
    idx = np.argmin(dists)
    i, j = edges[idx]
    return Haversine(dataset[i], dataset[j])

def orientation(p,q,r):
    """
    Return positive number if p-q-r are clockwise, neg if ccw, zero if colinear.
    """
    return (q[1]-p[1])*(r[0]-p[0]) - (q[0]-p[0])*(r[1]-p[1])

def hulls(dataset):
    """
    Graham scan to find upper and lower convex hulls of a set of 2d points.
    """
    U = []
    L = []
    dataset.sort()
    for p in dataset:
        while len(U) > 1 and orientation(U[-2],U[-1],p) <= 0: U.pop()
        while len(L) > 1 and orientation(L[-2],L[-1],p) >= 0: L.pop()
        U.append(p)
        L.append(p)
    return U,L

def rotating_calipers(dataset):
    """
    Given a list of 2d points, finds all ways of sandwiching the points between
    two parallel lines that touch one point each, and yields the sequence of pairs of
    points touched by each pair of lines.
    """
    U,L = hulls(dataset)
    i = 0
    j = len(L) - 1
    while i < len(U) - 1 or j > 0:
        yield U[i],L[j]
        
        # if all the way through one of top or bottom, advance the other
        if i == len(U) - 1: j -= 1
        elif j == 0: i += 1
        
        # still points left on both lists, compare slopes of next hull edges
        elif (U[i+1][1]-U[i][1])*(L[j][0]-L[j-1][0]) > \
                (L[j][1]-L[j-1][1])*(U[i+1][0]-U[i][0]):
            i += 1
        else: j -= 1

def max_distance(dataset):
    """
    Given a list of 2d points, returns the pair that is farthest apart.
    The key is to use the method of rotating calipers, an algorithm design
    technique that can be used to solve optimization problems including
    finding the width or diameter of a set of points (that is what we want).
    """
    best_dist = 0
    for p1, p2 in rotating_calipers(dataset):
        dist = Haversine(p1, p2)
        if dist > best_dist:
            best_dist = dist
    return best_dist


############################
# Computation of the betas #
############################

def compute_betas_for_fixed_eps(max_dist, min_dist, eps):
    betas = []
    i_min = int(math.ceil(math.log(min_dist)/math.log(1+eps)))
    i_max = int(math.floor(math.log(max_dist)/math.log(1+eps)))
    for i in range(i_min, i_max+1):
        betas.append((1+eps)**i)
    return betas

def compute_betas(max_dist, min_dist, epsilons):
    eps_to_betas = {}
    for eps in epsilons:
        betas = compute_betas_for_fixed_eps(max_dist, min_dist, eps)
        eps_to_betas[eps] = betas
        print('Number of betas for epsilon = ' + str(eps) + ' : ' + str(len(betas)))
    return eps_to_betas


#####################
# Static clustering #
#####################

def build_cluster(center_id, beta, dataset_ids, clusters, verbose):
    """
    This function takes the id of a cluster center, a certain beta, the list of
    point ids that are active within the dataset and still need to be clustered,
    and the dictionary of the clusters that have been built so far.
    
    It scans all the points that remain to be clustered and adds a point to the
    cluster if it is at a distance at most 2*beta from the center. 'Adding the
    point' to the cluster means mapping the point id to the center id within the
    clusters dictionnary (clusters), and removing the point from the list of
    points that have to be clustered (dataset_ids).
    """
    
    center = id_to_coords[center_id]    
    i = 0
    k = 0
    while True:
        if i >= len(dataset_ids):
            # we have scanned all the points that remained to be clustered
            break      
        k += 1
        point_id = dataset_ids[i]
        distance = Haversine(center, id_to_coords[point_id])
        if distance <= 2*beta:
            clusters[point_id] = (center_id)
            dataset_ids.remove(point_id)
        else:
            i += 1
            
    if verbose:
        print(str(k) + ' points have been scanned')
        print(str(len(dataset_ids)) +' points remain to be affected\n')
    return dataset_ids, clusters



def build_beta_clustering(k, beta, dataset_ids, verbose):
    """
    This function creates the whole clustering for a fixed valued of beta,
    by iterating the function build_clusters until k clusters have been found
    or until there are no more points to cluster. In the first case, the
    points that have not been clustered are labeled as outliers. The label
    for outliers is set at -1, which cannot be the id of any point (the ids
    are naturals). During the process, we keep track of the order of creation
    of the cluster (i.e. we store the centers ids in order). This will be useful
    in the dynamic version during the reclustering part.
    """
    clusters = {point_id : -1 for point_id in dataset_ids}
    # at the beginning, all points are labeled as outliers
    
    order = []
    d = deepcopy(dataset_ids)
    # we create a deep copy of the dataset such that the original one is preserved
    
    for i in range(0, k):
        if not d:
            break
        center_id = random.choice(d)
        # we pick u.a.r a not-yet-clustered point to build a new cluster around it
        
        order.append(center_id)
        # we keep track of the order of creation of the clusters through the center id
        
        d.remove(center_id)
        # the center no longer needs to be clustered !
        clusters[center_id] = center_id
        # the center belongs to its own cluster !

        if verbose:
            print('Building cluster ' + str(i+1) + ' whose center is ' + str(center_id))
        d, clusters = build_cluster(center_id, beta, d, clusters, verbose)
    return clusters, order


def build_betas_clustering(k, betas, dataset_ids, verbose):
    """
    This function takes a list of betas and computes a clustering for each
    beta of the list. It also computes, for each beta, the order of creation
    of the clusters (useful for the reclustering part).
    """
    clustering, ordering = {}, {}
    # these are dictionaries that map a value of beta to a clustering and an
    # ordering (which means an ordered list of the centers' ids).
    
    for beta in betas:
        if verbose:
            print('Computing for beta = ' + str(beta) + '\n')
        clusters, order  = build_beta_clustering(k, beta, dataset_ids, verbose)
        ordering[beta] = order
        clustering[beta] = clusters
    return clustering, ordering


def build_epsilons_clustering(k, eps_to_betas, dataset_ids, verbose):
    """
    This function computes a series of clustering for each epsilon
    (i.e. a clustering for beta associated to this epsilon). It also keeps 
    track of the clustering order, for each epsilon, for each beta.
    """
    times = []
    whole_clustering, whole_ordering = {}, {}
    
    for eps, betas in eps_to_betas.items():
        start = time.time()
        clustering, ordering = build_betas_clustering(k, betas, dataset_ids, verbose)
        whole_clustering[eps] = clustering
        whole_ordering[eps] = ordering
        end = time.time()
        times.append(end-start)
    return whole_clustering, whole_ordering, times

######################
# Dynamic clustering #
######################

def insertion(k, point_id, clustering, ordering, verbose):
    """
    This function takes a point and adds it to the differents clusterings
    maintained in parallel (that depend on epsilon and the related betas).
    The point is added to an existing cluster if it fits in a 2*beta
    neighborhood around a center (we consider the centers from the oldest
    to the newest). Otherwise, it becomes the center of its own cluster (if
    we have not reached the maximal number of clusters) or an outlier.
    """
    times = []
    point = id_to_coords[point_id]
    if verbose:
        print('New point to be added : ' + str(point_id))
    for epsilon, betas_clustering in clustering.items():
        start = time.time()
        for beta, beta_clustering in betas_clustering.items():
            if verbose:
                print('Inserting in the clustering of beta = ' + str(beta))
            if point_id in beta_clustering.keys():
                if verbose:
                    print('This point already exists : no changes')
                break
            centers_ids = ordering[epsilon][beta]
            found = 0
            for center_id in centers_ids:
                center = id_to_coords[center_id]
                if Haversine(center, point) <= 2*beta:
                    if verbose:
                        print('This point belongs to the cluster of point ' + str(center_id) + '\n')
                    beta_clustering[point_id] = center_id
                    found = 1
                    break
            if found == 0:
                if len(centers_ids) < k:
                    beta_clustering[point_id] = point_id
                    centers_ids.append(point_id)
                    if verbose:
                        print('This point will form a new cluster\n')
                else:
                    if verbose:
                        print('This point is not yet affected to any cluster\n')
                    beta_clustering[point_id] = -1
        end = time.time()
        times.append(end-start)
    return clustering, ordering, times


def deletion(point_id, clustering, ordering, verbose):
    """
    This function handles point deletion while maintaining the parzallel
    clustering (one clustering per value of beta, and several beta per value
    of epsilon...). If the point to be deleted is not a cluster center, we
    just remove it and leave the global clustering unchanged. If it is a
    center, we recluster all the points that  are affected to clusters
    "younger" than the cluster that has been destroyed.
    """
    times = []
    for epsilon, betas_clustering in clustering.items():
        start = time.time()
        for beta, beta_clustering in betas_clustering.items():
            if verbose:
                print('Deleting in the clustering of beta = ' + str(beta))
            if point_id not in beta_clustering.keys():
                if verbose:
                    print('This point does not exist : no changes')
                break
            beta_ordering = ordering[epsilon][beta]
            if point_id not in beta_ordering:
                if verbose:
                    print('This point was not a center, the deletion does not affect the whole clustering')
                del beta_clustering[point_id]
            else:
                if verbose:
                    print('This point is a center : reclustering all the clusters created after it')
                del beta_clustering[point_id]
                i = beta_ordering.index(point_id)
                # we capture the age of the cluster that has been destroy, to reconstruct
                # all the clusters that are younger than it
                if verbose:
                    total_n_clusters = len(beta_ordering)
                    print('This point was the center of the cluster number : ' + str(i))
                    print('We will have to recluster ' + str(total_n_clusters-i) + ' clusters')
                
                centers_to_recluster = beta_ordering[i:]
                beta_ordering = beta_ordering[:i]
                n_to_recluster = len(centers_to_recluster)
                centers_to_recluster.append(-1)
                # we will try to recluster all the outliers aas well !
                
                points_to_recluster = [p for p, c in beta_clustering.items() if c in centers_to_recluster]
                # we collect all the points that were affected to the "younger" clusters
                
                sub_clusters, sub_order = build_beta_clustering(n_to_recluster, beta, points_to_recluster, verbose)
                # we get a reclustering of the "younger" clusters only
                
                beta_clustering, beta_ordering = merge(beta_clustering, beta_ordering, sub_clusters, sub_order)
                # we merge the "older" clustering with the newer clustering
        end = time.time()
        times.append(end-start)
    return clustering, ordering, times


def merge(clusters, order, sub_clusters, sub_order):
    order = order + sub_order
    for point_id, center_id in sub_clusters.items():
        clusters[point_id] = center_id
    return clusters, order   






##################
# Sliding window #
##################

def sliding_window(window_width, n_operations, dataset_ids, k, eps_to_betas):
    clustering, ordering, times = build_epsilons_clustering(k, eps_to_betas, dataset_ids, False)
    print(times)
    for i in range(0, n_operations):
        j = i + window_width
        clustering, ordering, times_del = deletion(i, clustering, ordering, False)
        clustering, ordering, times_ins = insertion(k, j, clustering, ordering, False)
        times_ins_del = map(sum, zip(*[times_del, times_ins]))
        times = map(sum, zip(*[times, times_ins_del]))
    return times


if __name__ == '__main__':
    # we open the h5 file that contains the data
    f = h5py.File('dataset.hdf5', 'r')
    # we put the data into lists
    timestamps = f['timestamps'] # array of numpy ints
    latitudes = f['latitudes'] # list of numpy floats
    longitudes = f['longitudes'] # list of numpy floats
    dataset = list(zip(latitudes, longitudes)) # list of tuples (lat, lon)

    id_to_coords = {i : point for i, point in enumerate(dataset)}
    dataset_ids = id_to_coords.keys()

    epsilons = list(np.arange(0.1, 1.1, 0.1))
    eps_to_betas = compute_betas(max_dist, min_dist, epsilons)
    
    window_width = 60000
    n_operations = 100
    k = 20
    times = sliding_window(window_width, n_operations, dataset_ids, k, eps_to_betas)
    print(times, sum(times))