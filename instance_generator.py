# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 23:00:06 2019

@author: secan
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix


def generate_instance(y=1, p=5, s=3, v=6, n=18, save=True, directory='instances', plot=True):
    
    # Create Instances Folder
    if directory in os.listdir():
        os.chdir(directory)
    else:
        os.mkdir(directory)
        os.chdir(directory)
    
    # Create this instance folder
    
    n_instance = len([inst.replace('random', '') for inst in os.listdir() if 'random' in inst]) + 1
    os.mkdir('random' + str(n_instance))
    os.chdir('random' + str(n_instance))
    
    # Generate Network
    yards = np.random.uniform(-5000, 5000, size=(y, 2))
    sources = np.random.uniform(-5000, 5000, size=(n, 2))
    kmeans = KMeans(n_clusters=p)
    c = kmeans.fit_predict(sources)    
    pickups = kmeans.cluster_centers_
    shelters = np.random.uniform(-5000, 5000, size=(s, 2))
    
    N = np.concatenate([yards, pickups, shelters], axis=0)
    
    distances = distance_matrix(N, N)
    distances0 = distance_matrix(sources, pickups).min(axis=1)
    
    np.savetxt('nodes.txt', N)
    np.savetxt('source_nodes.txt', sources)
    np.savetxt('distances.txt', distances)
    np.savetxt('clusters.txt', c, fmt='%d')
    np.savetxt('distances0.txt', distances0)
    demands = np.ones(1)
    capacities = np.zeros(1)
    
    while demands.sum() > capacities.sum():
        # avoid infeasible instances
        demands = np.random.randint(5, 15, size=(n,))        
        capacities = np.random.randint(30, 50, size=(s,))
    
    np.savetxt('demands.txt', demands, fmt='%d')
    np.savetxt('capacities.txt', capacities, fmt='%d')
    
    buses = np.random.multinomial(4, np.ones(y)/y, size=(1))[0]
    np.savetxt('buses.txt', buses, fmt='%d')
    
    
    
    
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(sources[:, 0], sources[:, 1], color='teal', s=70, label='Source Nodes', zorder=3, alpha=0.5)
        ax.scatter(yards[:, 0], yards[:, 1], color='yellow', s=100, label='Bus Yards', zorder=3)
        ax.scatter(pickups[:, 0], pickups[:, 1], color='red', s=100, label='Pickup Points', zorder=3)
        ax.scatter(shelters[:, 0], shelters[:, 1], color='green', s=100, label='Shelters', zorder=3)
        
        for i in range(len(c)):
            ax.plot([sources[i, 0], pickups[c[i], 0]], [sources[i, 1], pickups[c[i], 1]], color='grey', linestyle='--', linewidth=1, alpha=0.5)
        # yards - pickups
        for i in range(y):
            for j in range(p):
                ax.plot([yards[i, 0], pickups[j, 0]], [yards[i, 1], pickups[j, 1]], color='#004b4f', linewidth=1)
        
        # fully connected network
        fc = np.concatenate([pickups, shelters], axis=0)
        
        for i in range(p + s):
            for j in range(p + s):
                if i != j:
                    ax.plot([fc[i, 0], fc[j, 0]], [fc[i, 1], fc[j, 1]], color='#004b4f', linewidth=1)
        ax.legend(frameon=False)
        ax.set_axis_off()
        ax.set_title('BEP: {} yards, {} pickups, {} shelters, {} buses'.format(y, p, s, v))
        
        fig.savefig('figure.png')
        plt.show()
        
        os.chdir('..')
    return N

generate_instance()
    