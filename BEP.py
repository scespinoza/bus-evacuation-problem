# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 23:49:11 2019

@author: secan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model





def read_instance(instance_name):
    
    nodes = np.loadtxt(instance_name + '/nodes.txt')
    distances = np.loadtxt(instance_name + '/distances.txt')
    distances0 = np.loadtxt(instance_name + '/distances0.txt')
    capacities = np.loadtxt(instance_name + '/capacities.txt', dtype='int32')
    demands = np.loadtxt(instance_name + '/demands.txt', dtype='int32')
    clusters = np.loadtxt(instance_name + '/clusters.txt', dtype='int32')
    buses = np.loadtxt(instance_name + '/buses.txt', dtype='int32')
    
    if buses.shape == ():
        buses = buses.reshape(1,)
    return nodes, distances, distances0, capacities, demands, buses, clusters




def BEP_MODEL(instance, objective='min-max', timelimit=4000, Q=20):
    
    if 'instances' in os.listdir():
        os.chdir('instances')
        
    nodes, distances, distances0, capacities, demands, buses, clusters = read_instance(instance)
    
    
    tau = distances / 16.666667
    # Sets
    
    Y = list(range(len(buses)))
    P = list(range(len(Y), len(Y) + len(np.unique(clusters))))
    S = list(range(len(Y) + len(P), len(Y) + len(P) + len(capacities)))
    
    
    N = Y + P + S
    N_prime = P + S
    
    A = [(i, j) for i in Y for j in P] + [(i, j) for i in (N_prime) for j in (N_prime) if i!=j]
    
    Vi = []
    for i in range(len(buses)):
        l  = len([item for sublist in Vi for item in sublist])
        Vi.append(list(range(l, l + int(buses[i]))))
        
    V = [bus for sublist in Vi for bus in sublist]
    
    D = dict()
    for i in range(len(P)):
        D[P[i]] = demands[clusters == i].sum()
        
    C = dict()
    for j in range(len(S)):
        C[S[j]] = capacities[j]
        
    T = list(range(1, 6 + 1))
    
    
    
    model = Model('BEP - ' + instance)
    
    x = {(i, j, m, t): model.binary_var(name='x_{}_{}_{}_{}'.format(i, j, m, t)) for i, j in A for m in V for t in T}
    b = {(j, m, t): model.integer_var(lb=0, name='b_{}_{}_{}'.format(j, m, t)) for j in N_prime for m in V for t in T}
    T_evac = model.continuous_var(lb=0, name='T_evac')
    
    if objective == 'min-max':
        
        model.minimize(T_evac)
        
    elif objective == 'cost':
        
        model.minimize(model.sum(model.sum(model.sum(tau[i, j] * x[i, j, m, t] for t in T) for m in V) for (i, j) in A))
        
        
    for m in V:
        
        model.add_constraint(T_evac >= model.sum(model.sum(tau[i, j] * x[i, j, m, t] for t in T) for i, j in A))
        
    for j in P:
        for m in V:
            for t in T[:-1]:
                
                model.add_constraint(model.sum(x[i, j, m, t] for i in N if (i, j) in A) == model.sum(x[j, k, m, t + 1] for k in N if (j, k) in A))
    
    for j in S:
        for m in V:
            for t in T[:-1]:
                
                model.add_constraint(model.sum(x[i, j, m, t] for i in N if (i, j) in A) >= model.sum(x[j, k, m, t + 1] for k in N if (j, k) in A))
                
    for m in V:
        for t in T:
            
            model.add_constraint(model.sum(x[i, j, m, t] for (i, j) in A) <= 1)
        
    for i in Y:
        for m in Vi[i]:
                    
            model.add_constraint(model.sum(x[i, j, m, 1] for j in P) == 1)
                    
    for i in Y:
        for j in N:
            if (i, j) in A:
                for m in V:
                    for t in T[1:]:
                        
                        model.add_constraint(x[i, j, m, t] == 0)
                        
    for j in P:
        for i in N:
            if (i, j) in A:
                for m in V:
                    
                    model.add_constraint(x[i, j, m, 6] == 0)
                    
    
    for j in N_prime:
        for m in V:
            for t in T:
                
                model.add_constraint(b[j, m, t] <= model.sum(Q * x[i, j, m, t] for i in N if (i, j) in A))
                
    for m in V:
        for t in T:
            
            model.add_constraint(0 <= model.sum(model.sum(b[j, m, l] for l in range(1, t + 1)) for j in P) - model.sum(model.sum(b[k, m, l] for l in range(1, t + 1)) for k in S))
            model.add_constraint(model.sum(model.sum(b[j, m, l] for l in range(1, t + 1)) for j in P) - model.sum(model.sum(b[k, m, l] for l in range(1, t + 1)) for k in S) <= Q)
    
    for j in S:
        model.add_constraint(model.sum(model.sum(b[j, m, t] for t in T) for m in V) <= C[j])    
        
    for j in P:
        model.add_constraint(model.sum(model.sum(b[j, m, t] for t in T) for m in V) == D[j])
        
    
    for m in V:
        
        model.add_constraint(model.sum(model.sum(b[j, m, t] for t in T) for j in P) == model.sum(model.sum(b[k, m, t] for t in T) for k in S))
        
        
    model.set_time_limit(timelimit)
    print('Solving...')
    
    model.solve()
    
    if not model.solution:
        print('not solution')
        return model
    
    ###################################################
    # Output Files
    ###################################################
            
    
    def plot_graph():
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(nodes[Y, 0], nodes[Y, 1], color='yellow', s=100, label='Bus Yards', zorder=3)
        ax.scatter(nodes[P, 0], nodes[P, 1], color='red', s=100, label='Pickup Points', zorder=3)
        ax.scatter(nodes[S, 0], nodes[S, 1], color='green', s=100, label='Shelters', zorder=3)
        
        # yards - pickups
        for i in range(len(Y)):
            for j in range(len(P)):
                ax.plot([nodes[Y][i, 0], nodes[P][j, 0]], [nodes[Y][i, 1], nodes[P][j, 1]], color='#004b4f', linewidth=1, alpha=0.1)
        
        # fully connected network
        fc = np.concatenate([nodes[P], nodes[S]], axis=0)
        
        for i in range(len(P) + len(S)):
            for j in range(len(P) + len(S)):
                if i != j:
                    ax.plot([fc[i, 0], fc[j, 0]], [fc[i, 1], fc[j, 1]], color='#004b4f', linewidth=1, alpha=0.1)
        ax.legend(frameon=False)
        ax.set_axis_off()
        ax.set_title('BEP: {} yards, {} pickups, {} shelters, {} buses'.format(len(Y), len(P), len(S), len(V)))
        
        return fig, ax
    
    
    
    os.chdir(instance)
    if not 'solution_bep' in os.listdir():
        os.mkdir('solution_bep')
    
    
    routes = {}
    
    for m in V:
        
        fig, ax = plot_graph()
        routes[m] = []
        
        for t in T:
            
            for (i, j) in A:
                if x[i, j, m, t].solution_value == 1:
                    ax.plot(nodes[(i, j), 0], nodes[(i, j), 1], color='green', linewidth=2)
                    if j in P:
                        ax.annotate('{:.0f} Pickups\nTotal demand: {:.0f}'.format(b[j, m, t].solution_value, D[j]), nodes[j],
                                    horizontalalignment='center')
                    routes[m].append((i, j))   
                
        ax.set_title('Route {}'.format(m))
        fig.savefig('solution_bep/{}-route{}.png'.format(objective, m))
        
    # compute total cost    
    total_cost = 0
    for (i, j) in A:
        for m in V:
            for t in T:
                total_cost += tau[i, j] * x[i, j, m, t].solution_value
                
    bus_times = []
    for m in V:
        bus_time = 0
        for (i, j) in A:
            for t in T:
                bus_time += tau[i, j] * x[i, j, m, t].solution_value
        bus_times.append(bus_time)
        
    with open('solution_bep/sol_{}.txt'.format(objective), 'w') as out:
        out.write(str(model.solution))
        out.write('\n' + '#'*20)
        out.write(' Solve details ')
        out.write('#'*20 + '\n')
        out.write(str(model.solve_details))
        out.write('#'*20 + '\n')
        out.write('Total Cost: {}\n'.format(total_cost))
        out.write('Evacuation Time: {}'.format(max(bus_times)))
    
    print('return')
    os.chdir('..')
    return model
        
if __name__ == '__main__':
    
    
    model = BEP_MODEL('paipote', objective='cost')
    model = BEP_MODEL('paipote', objective='min-max')

                
    