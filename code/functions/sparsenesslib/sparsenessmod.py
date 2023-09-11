#!/usr/bin/env python
#!/usr/bin/env python
#####################################################################################
# DESCRIPTION:
#####################################################################################
#Hard to classify functions, mainly manipulation and conversion of data structures

#1. parse_rates: Stores notes and image names contained in *labels_path* in *dict_labels* as {name:note} 

#2. create_dataframe: Creates a pandas dataframe that has a beauty score associates the metric of the associated image layers

#####################################################################################
# LIBRAIRIES:
#####################################################################################
#public librairies
import csv
import pandas
#####################################################################################
# PROCEDURES/FUNCTIONS:
#####################################################################################
def parse_rates(labels_path , dict_labels):
    '''
    Stores notes and image names contained in *labels_path* 
    in *dict_labels* as {name:note}    
    '''
    with open(labels_path, newline='') as labels:
        reader = csv.reader(labels)
        for line in reader:
            key = line[0]
            rate = line[1]
            dict_labels[key] = float(rate)
#####################################################################################
def create_dataframe(dict_rates, dict_metric, name = 'rate'):
    '''
    Creates a pandas dataframe that has a beauty score associates
    the metric of the associated image layers
    rows: images, column 1: beauty rate, column 2 to n: metric
    '''
    df1 = pandas.DataFrame.from_dict(dict_rates, orient='index', columns = [name])
    df2 = pandas.DataFrame.from_dict(dict_metric, orient='index')     

    df = pandas.concat([df1, df2], axis = 1)     
    return df



