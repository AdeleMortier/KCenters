import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import numpy as np

def plot_times(n_operations, k, window_width):
    folder_name = str(k) + '_' + str(n_operations) + '_' + str(window_width)
    f = open('files/' + folder_name + '/times_cumulated_'+ folder_name +'.txt')
    times_per_eps = defaultdict(list)
    for j, line in enumerate(f):
        if j == 0:
            epsilons = line.split("    ")
            epsilons = epsilons[1:]
            epsilons = [float(eps) for eps in epsilons]
        else:
            times = line.split("    ")
            times = times[1:]
            times = [float(time) for time in times]
            for i, time in enumerate(times):
                times_per_eps[i].append(time)
    n_eps = len(epsilons)
    colors = [(np.random.uniform(0, 1, 3)) for _ in range(0, n_eps)]
    for i, times in times_per_eps.items():
        plt.plot(times, label=("eps = " + str(epsilons[i])), color=colors[i])
    plt.legend()
    plt.title("Cumulated execution time at each step for each epsilon,\n" + 
              str(k) + ' clusters, ' + str(n_operations) +
              ' operations, window of ' + str(window_width) + ' tweets')
    plt.savefig("files/" + folder_name + '/cumulated_execution_times_' + folder_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()

    final_times = [times[-1] for times in times_per_eps.values()]
    epsilons, final_times = list(zip(*sorted(zip(epsilons, final_times))))
    plt.plot(epsilons, final_times)
    plt.title("Total execution time as a function of epsilon\n" + 
              str(k) + ' clusters, ' + str(n_operations) +
              ' operations, window of ' + str(window_width) + ' tweets')
    plt.savefig('files/'+ folder_name + '/final_execution_times_' + folder_name + '.png')
    plt.clf()
    plt.cla()
    plt.close()

    

    final_times_updates_only = [(times[-1]-times[0])/n_operations for times in times_per_eps.values()]
    epsilons, final_times = list(zip(*sorted(zip(epsilons, final_times_updates_only))))
    plt.plot(epsilons, final_times)
    plt.title("Average execution time for an update\n" + 
              str(k) + ' clusters, ' + str(n_operations) +
              ' operations, window of ' + str(window_width) + ' tweets')
    plt.savefig('files/' + folder_name + '/average_execution_times_per_update_' + folder_name + ".png")
    plt.clf()
    plt.cla()
    plt.close()

    final_times = list(final_times)
    print(final_times)
    epsilons= list(epsilons)
    epsilons = [str(epsilon) for epsilon in epsilons]
    runtime_ratios = [final_times[i+1]/final_times[i] for i in range(0, len(final_times)-1)]
    runtime_ratios = [str(runtime_ratio) for runtime_ratio in runtime_ratios]
    f = open('files/' + folder_name + '/runtime_ratios.txt', 'w')
    f.write("    ".join(epsilons))
    f.write('\n')
    f.write("    ".join(runtime_ratios))


    

def plot_clusters_on_map(id_to_coords, col, clusters, order, number):
    centers_ids = order
    n_clusters = len(clusters.keys())

    fig = plt.figure(figsize=(8, 10), dpi=100)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.stock_img()
    ax.coastlines()
    for center_id in centers_ids:
        affected_points = [point_id for point_id, c in clusters.items() if c == center_id]
        affected_points = random.sample(affected_points, min(100, len(affected_points)))
        latitudes = [id_to_coords[i][0] for i in affected_points]
        longitudes = [id_to_coords[i][1] for i in affected_points]
        c = col[centers_ids.index(center_id)]
        plt.plot(latitudes, longitudes, '+', transform=ccrs.PlateCarree(), color=c)
        center_latitude = id_to_coords[center_id][0]
        center_longitude = id_to_coords[center_id][1]
        plt.plot(center_latitude, center_longitude, 'o', transform=ccrs.PlateCarree(), color=c, markeredgewidth=1., markeredgecolor='black')
    plt.title('Cluster samples (round=centers; cross=random points)')
    fig.savefig('files/map_' + str(k) + '_' + str(n_operations) + '_' + str(window_width) + '_' + str(number) + '.png')
    plt.clf()
    plt.cla()
    plt.close()








if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k',
                        type=int,
                        default='15')

    parser.add_argument('--n_operations', type=int, default='60000')
    parser.add_argument('--window_width', type=int, default='10000')
    args = parser.parse_args()

    window_width = args.window_width
    n_operations = args.n_operations
    k = args.k

    print("Plotting...")
    plot_times(n_operations, k, window_width)




