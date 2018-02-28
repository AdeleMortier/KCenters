import math
from scipy import spatial
import numpy as np


#########################
# Min and max distances #
#########################

def Haversine(point, neighbor):
    """
    The function takes two tuples (two points) and computes their Haversine
    distance.
    """
    lat1 = point[0]
    lat2 = neighbor[0]
    lon1 = point[1]
    lon2 = neighbor[1]
    R = 6371000
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2.0)**2 + math.cos(phi_1) * math.cos(phi_2) * \
        math.sin(delta_lambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R*c/1000


def min_distance(dataset):
    """
    This function computes the distance between the two closest points of the
    dataset. It is based on the Delaunay triangulation (by definition the
    closest pair of points in a set define an edge in the Delaunay
    triangulation of this set). We adapted a bit the original version (that
    relies on the Euclidean distance), to consider instead the Haversine
    distance.
    """
    # set up the triangulation
    mesh = spatial.Delaunay(dataset)
    edges = np.vstack((mesh.vertices[:, :2], mesh.vertices[:, -2:]))
    points1 = mesh.points[edges[:, 0]]
    points2 = mesh.points[edges[:, 1]]

    # here we adapted the code to use the Haversine distance instead of the
    # Euclidean distance
    dists = ([Haversine(p1, p2) for p1, p2 in zip(points1, points2)])
    idx = np.argmin(dists)
    i, j = edges[idx]
    return Haversine(dataset[i], dataset[j])


def orientation(p, q, r):
    """
    Return positive number if p-q-r are clockwise, neg if ccw, zero if
    colinear.
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
        while len(U) > 1 and orientation(U[-2], U[-1], p) <= 0:
            U.pop()
        while len(L) > 1 and orientation(L[-2], L[-1], p) >= 0:
            L.pop()
        U.append(p)
        L.append(p)
    return U, L


def rotating_calipers(dataset):
    """
    Given a list of 2d points, finds all ways of sandwiching the points between
    two parallel lines that touch one point each, and yields the sequence of
    pairs of points touched by each pair of lines.
    """
    U, L = hulls(dataset)
    i = 0
    j = len(L) - 1
    while i < len(U) - 1 or j > 0:
        yield U[i], L[j]

        # if all the way through one of top or bottom, advance the other
        if i == len(U) - 1:
            j -= 1
        elif j == 0:
            i += 1

        # still points left on both lists, compare slopes of next hull edges
        elif (U[i+1][1]-U[i][1])*(L[j][0]-L[j-1][0]) > \
                (L[j][1]-L[j-1][1])*(U[i+1][0]-U[i][0]):
            i += 1
        else:
            j -= 1


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
        print('Number of betas for epsilon = ' + str(eps) + ' : ' +
              str(len(betas)))
    return eps_to_betas
