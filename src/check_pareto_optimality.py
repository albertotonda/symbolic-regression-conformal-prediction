# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:29:37 2024

Simple and inefficient script to check the Pareto optimality between
a few methods (so, just a few points in a 2D space).

@author: Alberto
"""

import pandas as pd
import sys


def is_dominated(considered_point, other_point) :
    
    # first fitness is median, smaller is better; second fitness is coverage, larger is better
    if considered_point[0] > other_point[0] and considered_point[1] <= other_point[1] :
        return True
    else :
        return False


if __name__ == "__main__" :
    
    # hard-coded values
    fitness_1_name = "_coverage"
    fitness_2_name = "_median"
    
    results_file = "../results/2024-06-07-full-results/results.csv"
    df = pd.read_csv(results_file)
    
    # find the names for all methods, and the corresponding columns
    methods = dict()
    
    # first, get all the columns that either end with fitness_1 or fitness_2
    fitness_columns = [c for c in df.columns if c.endswith(fitness_1_name) or c.endswith(fitness_2_name)]
    
    # then, iterate over the columns and get names
    for fc in fitness_columns :
        if fc.find(fitness_1_name) != -1 :
            method = fc[:-len(fitness_1_name)]
        elif fc.find(fitness_2_name) != -1 :
            method = fc[:-len(fitness_2_name)]
            
        if method not in methods :
            methods[method] = []
        
        methods[method].append(fc)
        
    # data structure to collect the final statistics
    methods_statistics = { method : {"non-dominated" : 0, "dominated" : 0, "alone" : 0} for method in methods }
    
    # now, we iterate over all rows (representing performance on a data set),
    # and mark the pareto optimality of each method
    for index, row in df.iterrows() :
        
        print("Now analyzing dataset \"%s\"..." % row["dataset_name"])
        non_dominated_methods = []
        
        # generate the set of points          
        points = [(row[methods[method][0]], row[methods[method][1]]) for method in methods]
        #print(points)
        if row["dataset_name"] == "red_wine" :
            for i, method in enumerate(methods) :
                print("Method: %s, point: (%s)" % (method, str(points[i])))
        
        # now, check if the point corresponding to a method is dominated
        for method in methods :
            method_point = (row[methods[method][0]], row[methods[method][1]])
            #print("Method %s, point: %s" % (method, str(method_point)))
            
            is_method_point_dominated = False
            for point in points :
                if is_dominated(method_point, point) :
                    is_method_point_dominated = True
                
            if is_method_point_dominated == False :
                non_dominated_methods.append(method)
                methods_statistics[method]["non-dominated"] += 1
            else :
                methods_statistics[method]["dominated"] += 1
            
        print("Non-dominated methods:", non_dominated_methods)
        if len(non_dominated_methods) == 1 :
            methods_statistics[non_dominated_methods[0]]["alone"] += 1
            
    # and now, some nice formatting for the results
    df_statistics = pd.DataFrame.from_dict(methods_statistics)
    df_statistics.to_csv("results-statistics.csv")