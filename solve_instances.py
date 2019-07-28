# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 00:22:17 2019

@author: secan
"""

from BEP import BEP_MODEL
from MABEP import MABEP_MODEL


instances = [
        #'random1',
         #    'random2',
          #   'random3',
             'paipote']

objectives = ['min-max', 'cost']


models_bep = {}
models_mabep = {}
for instance in instances:
    for obj in objectives:
        
        if instance == 'paipote':
            
            print('Solving {} - BEP, {} objective'.format(instance, obj))
            models_bep[instance, obj] = BEP_MODEL(instance, objective=obj, Q=30)
            
            print('Solving {} - MABEP, {} objective'.format(instance, obj))
            models_mabep[instance, obj] = MABEP_MODEL(instance, objective=obj, Q=30)
            
        else:
            print('Solving {} - BEP, {} objective'.format(instance, obj))
            models_bep[instance, obj] = BEP_MODEL(instance, objective=obj, Q=20)
            
            print('Solving {} - MABEP, {} objective'.format(instance, obj))
            models_mabep[instance, obj] = MABEP_MODEL(instance, objective=obj, Q=20)
        
