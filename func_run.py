import func_wffpa as wf
import numpy as np
import os
import time
import run_wffpa

def run_heatmap_g(sets,data,map_type):
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------','\n')
    print(' This will generate heatmaps for the following sets:')
    for i in range(0,len(sets),2):
        print(round(sets[i]),'-',round(sets[i+1]))
    save = True
    usr = input(' Is the fuel evaluated in these tests live or oven dried? (L/O/b)')
    if usr == 'l':
        thresh = 35
    elif usr == 'b':
        return
    elif usr == 'o':
        thresh = 50
    else:
        print('Error')
    func = run_wffpa.loop_handl(sets,data,'getmap',[thresh,save,map_type])
    return func

def run_heatmap_d(sets,heat_maps,data,cmap):
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------','\n')
    print(' This will display heatmaps for the following test numbers:')
    for i in range(0,len(sets),2):
        print(round(sets[i]),'-',round(sets[i+1]))

    usr = input('Would you like to display heatmaps for the complete test, pre-ignition, or ignition? (c/p/i): ')
    if usr == 'c':
        map_type = 'all'
    elif usr == 'p':
        map_type = 'preig'
    elif usr == 'i':
        map_type = 'ig'
    else:
        return
    run_wffpa.loop_handl(sets,data,'displaymap',[map_type,cmap])
    return

def run_selectpoints_s(sets,data):
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------','\n')
    print(' This will allow the user to select points for the following sets:')
    for i in range(0,len(sets),2):
        print(round(sets[i]),'-',round(sets[i+1]))
    usr = input('Select points for flame profiles, flame timeline, vertical profile, or grid? (hit \'b\' to go back (p/t/v/g): ')
    if usr == 'p':
        points_type = 'profile'
    elif usr == 't':
        points_type = 'timeline'
    elif usr == 'v':
        points_type = 'vert'
    elif usr == 'g':
        points_type = 'grid'
    else:
        return
    run_wffpa.loop_handl(sets,data,'selectpoints',args=[points_type])


def run_linedisplay_c(test,distance):
    usr = input('Continue (\'b\' to go back)')
    if usr == 'b':
        return 999
    coordinates = wf.get_line_coordinates(test,distance)
    heatmap = wf.load_heatmap(test)
    iscoordinates = isinstance(coordinates,np.ndarray)
    isheatmap = isinstance(heatmap,np.ndarray)
    if isheatmap == 0 or iscoordinates == 0:
        return None
    wf.display_linedisplay(heatmap,coordinates[0,:])
    return None
