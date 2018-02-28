from __future__ import generators
import cartopy.crs as ccrs 
from copy import deepcopy
import h5py
import numpy as np
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
import sys
import os
import logging as log


import preprocessing as pre
import postprocessing as post


#####################
# Static clustering #
#####################

def build_cluster(id_to_coords, center_id, beta, dataset_ids, clusters):
    """
    This function takes the id of a cluster center, a certain beta, the list of
    point ids that are active within the dataset and still need to be
    clustered, and the dictionary of the clusters that have been built so far.

    It scans all the points that remain to be clustered and adds a point to the
    cluster if it is at a distance at most 2*beta from the center. 'Adding the
    point' to the cluster means mapping the point id to the center id within
    the clusters dictionnary (clusters), and removing the point from the list
    of points that have to be clustered (dataset_ids).
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
        distance = pre.Haversine(center, id_to_coords[point_id])
        if distance <= 2*beta:
            clusters[point_id] = (center_id)
            dataset_ids.remove(point_id)
        else:
            i += 1

    log.info(str(k) + ' points have been scanned')
    log.info(str(len(dataset_ids)) + ' points remain to be affected\n')
    return dataset_ids, clusters


def build_beta_clustering(id_to_coords, k, beta, dataset_ids):
    """
    This function creates the whole clustering for a fixed valued of beta, by
    iterating the function build_clusters until k clusters have been found or
    until there are no more points to cluster. In the first case, the points
    that have not been clustered are labeled as outliers. The label for
    outliers is set at -1, which cannot be the id of any point (the ids are
    naturals). During the process, we keep track of the order of creation of
    the cluster (i.e. we store the centers ids in order). This will be useful
    in the dynamic version during the reclustering part.
    """
    clusters = {point_id: -1 for point_id in dataset_ids}
    # at the beginning, all points are labeled as outliers

    order = []
    d = deepcopy(dataset_ids)
    # we create a deep copy of the dataset such that the original one is
    # preserved

    for i in range(0, k):
        if not d:
            break
        center_id = random.choice(d)
        # we pick u.a.r a not-yet-clustered point to build a new cluster around
        # it

        order.append(center_id)
        # we keep track of the order of creation of the clusters through the
        # center id

        d.remove(center_id)
        # the center no longer needs to be clustered !
        clusters[center_id] = center_id
        # the center belongs to its own cluster !

        log.info('Building cluster ' + str(i+1) + ' whose center is ' +
                  str(center_id))
        d, clusters = build_cluster(id_to_coords, center_id,
                                    beta, d, clusters)
    return clusters, order


def build_betas_clustering(id_to_coords, k, betas, dataset_ids):
    """
    This function takes a list of betas and computes a clustering for each
    beta of the list. It also computes, for each beta, the order of creation
    of the clusters (useful for the reclustering part).
    """
    clustering, ordering = {}, {}
    # these are dictionaries that map a value of beta to a clustering and an
    # ordering (which means an ordered list of the centers' ids).

    for beta in betas:
        log.info('Computing for beta = ' + str(beta) + '\n')
        clusters, order = build_beta_clustering(id_to_coords, k, beta,
                                                dataset_ids)
        ordering[beta] = order
        clustering[beta] = clusters
    return clustering, ordering


def build_epsilons_clustering(id_to_coords, k, eps_to_betas, dataset_ids):
    """
    This function computes a series of clustering for each epsilon
    (i.e. a clustering for beta associated to this epsilon). It also keeps
    track of the clustering order, for each epsilon, for each beta.
    """
    times = {}
    whole_clustering, whole_ordering = {}, {}

    for eps, betas in eps_to_betas.items():
        start = time.time()
        clustering, ordering = build_betas_clustering(id_to_coords, k, betas,
                                                      dataset_ids)
        whole_clustering[eps] = clustering
        whole_ordering[eps] = ordering
        end = time.time()
        times[eps] = (end-start)
    return whole_clustering, whole_ordering, times

def best_betas(whole_clustering, k, n_operations, window_width):
    folder_name = str(k) + '_' + str(n_operations) + '_' + str(window_width)
    f = open('files/' + folder_name + '/best_betas_'+folder_name+'.txt', 'w')
    best_of_all = 0
    for eps, betas_clustering in whole_clustering.items():
        beta_candidates = []
        for beta, beta_clustering in betas_clustering.items():
            if -1 not in beta_clustering.values():
                beta_candidates.append(beta)
        best_beta = min(beta_candidates)
        if eps == 0.1:
            best_of_all = best_beta
        f.write(str(eps) + "    " + str(best_beta) + '\n')
    f.close()
    return best_of_all



######################
# Dynamic clustering #
######################


def insertion(id_to_coords, k, point_id, clustering, ordering):
    """
    This function takes a point and adds it to the differents clusterings
    maintained in parallel (that depend on epsilon and the related betas).
    The point is added to an existing cluster if it fits in a 2*beta
    neighborhood around a center (we consider the centers from the oldest to
    the newest). Otherwise, it becomes the center of its own cluster (if we
    have not reached the maximal number of clusters) or an outlier.
    """
    times = {}
    point = id_to_coords[point_id]
    log.info('New point to be added : ' + str(point_id))
    for epsilon, betas_clustering in clustering.items():
        start = time.time()
        for beta, beta_clustering in betas_clustering.items():
            log.info('Inserting in the clustering of beta = ' + str(beta))
            if point_id in beta_clustering.keys():
                log.info('This point already exists : no changes')
                break
            centers_ids = ordering[epsilon][beta]
            found = 0
            for center_id in centers_ids:
                center = id_to_coords[center_id]
                if pre.Haversine(center, point) <= 2*beta:
                    log.info('This point belongs to the cluster of point ' +
                              str(center_id) + '\n')
                    beta_clustering[point_id] = center_id
                    found = 1
                    break
            if found == 0:
                if len(centers_ids) < k:
                    beta_clustering[point_id] = point_id
                    centers_ids.append(point_id)
                    log.info('This point will form a new cluster\n')
                else:
                    log.info('This point is not yet affected to any cluster\n')
                    beta_clustering[point_id] = -1
        end = time.time()
        times[epsilon] = (end-start)
    return clustering, ordering, times


def deletion(id_to_coords, point_id, clustering, ordering, n_op, col):
    """
    This function handles point deletion while maintaining the parzallel
    clustering (one clustering per value of beta, and several beta per value of
    epsilon...). If the point to be deleted is not a cluster center, we just
    remove it and leave the global clustering unchanged. If it is a center, we
    recluster all the points that  are affected to clusters "younger" than the
    cluster that has been destroyed.
    """
    times = {}
    reclustered = False
    # reclustered is a flag that says if we have reclustered a subset of clusters during the current call to deletion

    for epsilon, betas_clustering in clustering.items():
        start = time.time()
        for beta, beta_clustering in betas_clustering.items():
            log.info('Deleting in the clustering of beta = ' + str(beta))
            if point_id not in beta_clustering.keys():
                log.info('This point does not exist : no changes')
                break
            beta_ordering = ordering[epsilon][beta]
            if point_id not in beta_ordering:
                log.info('This point was not a center, the deletion does' +
                          ' not affect the whole clustering')
                del beta_clustering[point_id]
            else:
                log.info('This point is a center : reclustering all the ' +
                          'clusters created after it')
                reclustered = True
                del beta_clustering[point_id]
                i = beta_ordering.index(point_id)
                # we capture the age of the cluster that has been destroyed ,
                # to reconstruct all the clusters that are younger than it
                total_n_clusters = len(beta_ordering)
                log.info('This point was the center of the cluster number: ' +
                          str(i))
                log.info('We will have to recluster ' +
                          str(total_n_clusters-i) + ' clusters')

                centers_to_recluster = beta_ordering[i:]
                beta_ordering = beta_ordering[:i]
                n_to_recluster = len(centers_to_recluster)
                centers_to_recluster.append(-1)
                # we will try to recluster all the outliers as well !

                points_to_recluster = [p for p, c in beta_clustering.items()
                                       if c in centers_to_recluster]
                # we collect all the points that were affected to the "younger"
                # clusters

                sub_clusters, sub_order = build_beta_clustering(id_to_coords,
                                                                n_to_recluster,
                                                                beta,
                                                                points_to_recluster)
                # we get a reclustering of the "younger" clusters only

                beta_clustering, beta_ordering = merge(beta_clustering,
                                                       beta_ordering,
                                                       sub_clusters,
                                                       sub_order)
                # we merge the "older" clustering with the newer clustering
        end = time.time()
        times[epsilon] = (end-start)
    return clustering, ordering, times, reclustered


def merge(clusters, order, sub_clusters, sub_order):
    order = order + sub_order
    for point_id, center_id in sub_clusters.items():
        clusters[point_id] = center_id
    return clusters, order

def sort_times(times):
    epsilons_and_times = sorted([(eps, time) for eps, time in times.items()])
    epsilons, times = zip(*epsilons_and_times)
    epsilons_str = [str(eps) for eps in epsilons]
    times_str = [str(time) for time in times]
    epsilons_str = "    ".join(epsilons_str)
    times_str = "    ".join(times_str)
    return times, epsilons_str, times_str

##################
# Sliding window #
##################

def sliding_window(id_to_coords, window_width, n_operations,
                   dataset_ids, k, eps_to_betas):
    folder_name = str(k) + '_' + str(n_operations) + '_' + str(window_width)
    if not os.path.exists('files/' + folder_name):
        os.makedirs('files/' + folder_name)
    else:
        print('Folder for these parameters already exists, please delete if before')
        sys.exit(0)
    f = open('files/' + folder_name + '/times_cumulated_'+folder_name+'.txt', 'w')
    f2 = open('files/' + folder_name + '/best_beta_' + folder_name + '.txt', 'w')
    col = [(np.random.uniform(0, 1, 3)) for _ in range(0, k)]

    seed = dataset_ids[:window_width]
    clustering, ordering, times = build_epsilons_clustering(id_to_coords, k,
                                                            eps_to_betas,
                                                            seed)
    _, epsilons_str, times_str = sort_times(times)
    f.write('Epsilons    ' + epsilons_str + '\n')
    f.write('First clustering    ' + times_str + '\n')
    best_of_all = best_betas(clustering, k, n_operations, window_width) 
    f2.write(str(best_of_all) + '\n')
    print('Beginning adversarial clustering')
    for i in range(0, n_operations):
        j = i + window_width
        clustering, ordering, times_del, reclustered = deletion(id_to_coords, i, clustering,
                                                   ordering, i, col)
        clustering, ordering, times_ins = insertion(id_to_coords, k, j,
                                                    clustering, ordering)
        times_ins, _, _ = sort_times(times_ins)
        times_del, _, _ = sort_times(times_del)

        times_ins_del = map(sum, zip(*[times_del, times_ins]))
        new_times = map(sum, zip(*[times, times_ins_del]))
        if (i+1)%500 == 0 and i != 0:
            print('Operation ' + str(i))
            f.write('Operation ' + str(i) + '    ' + '    '.join([str(time) for time in new_times]) + '\n')
            best_of_all = best_betas(clustering, k, n_operations, window_width) 
            f.write('Operation' + str(i) + '    ' + str(best_of_all) + '\n')
        times = new_times
    f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k',
                        type=int,
                        default='15')

    parser.add_argument('--n_operations', type=int, default='60000')
    parser.add_argument('--window_width', type=int, default='10000')
    parser.add_argument('--verbose', type=bool, default=False)


    args = parser.parse_args()
    if args.verbose:
        log.basicConfig(format='%(levelname)s: %(message)s', level=log.DEBUG)
        log.info('Verbose output.')
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    window_width = args.window_width
    n_operations = args.n_operations
    k = args.k



    print('Preprocessing...')

    if 'files' not in os.listdir('.'):
        os.makedirs('files')

    # we open the h5 file that contains the data
    f = h5py.File('dataset.hdf5', 'r')
    # we put the data into lists
    timestamps = f['timestamps'][:window_width+n_operations]  # array of numpy ints
    latitudes = f['latitudes'][:window_width+n_operations]  # list of numpy floats
    longitudes = f['longitudes'][:window_width+n_operations]  # list of numpy floats
    dataset = list(zip(latitudes, longitudes))  # list of tuples (lat, lon)

    id_to_coords = {i: point for i, point in enumerate(dataset)}
    dataset_ids = id_to_coords.keys()

    epsilons = list(np.arange(0.1, 1.1, 0.1))
    max_dist = pre.max_distance(dataset)
    min_dist = pre.min_distance(np.asarray(dataset))
    eps_to_betas = pre.compute_betas(max_dist, min_dist, epsilons)    
    print('Computing clustering...')
    times = sliding_window(id_to_coords, window_width, n_operations,
                           dataset_ids, k, eps_to_betas)


    print('Plotting...')
    post.plot_times(n_operations, k, window_width)


