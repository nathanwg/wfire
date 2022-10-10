import func_wffpa as wf
import numpy as np
import os
import time

def run_heatmap_g(sets,data):
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------','\n')
    print(' This will generate heatmaps for the following sets:')
    for i in range(0,len(sets),2):
        print(round(sets[i]),'-',round(sets[i+1]))
    save = False
    usr = input(' Is the fuel evaluated in these tests live or oven dried? (L/O/b)')
    if usr == 'l':
        thresh = 35
    elif usr == 'b':
        return
    elif usr == 'o':
        thresh = 50
    else:
        print('Error')
    for i in range(0,len(sets),2):
        tests = np.linspace(int(sets[i]),int(sets[i+1]),int(sets[i+1]-sets[i]+1))
        heat_maps = wf.get_heatmaps(tests,data,False,True,thresh)
    return heat_maps

def run_heatmap_d(sets,heat_maps,data):
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------','\n')
    print(' This will display heatmaps for the following test numbers:')
    for i in range(0,len(sets),2):
        print(round(sets[i]),'-',round(sets[i+1]))
##    usr = input(' Would you like to display a custom set? (y/n/b)')
##    if usr == 'y':
##        sets = wf.change_tests()
##    elif usr == 'n':
##        print()
##    elif usr == 'b':
##        return
##    os.system('cls')
##    print('\n\n\n\n\n','--------------------------------------------------','\n')
    usr = input( 'Would you like to display from current loaded data or saved data? (c/s/b)')
    if usr == 'c':
        if heat_maps is None:
            print('There is no current loaded data')
            input('')
        else:
            wf.displaymaps(heat_maps,sets[0],sets[-1])
    elif usr == 's':
        for i in range(0,len(sets),2):
            tests = np.linspace(int(sets[i]),int(sets[i+1]),int(sets[i+1]-sets[i]+1))
            heat_maps = wf.load_heatmaps(tests,data)
            running = wf.displaymaps(heat_maps,sets[i],sets[i+1])
            if running is False:
                return
    elif usr == 'b':
        return
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
    for i in range(0,len(sets),2):
        tests = np.linspace(int(sets[i]),int(sets[i+1]),int(sets[i+1]-sets[i]+1))
        heat_maps = wf.load_heatmaps(tests,data)
        loop_count = 0
        for j in heat_maps:
            points,num_points,isvalid = wf.get_points(j,data,tests[loop_count],points_type)
            if isvalid is False:
                loop_count += 1
                continue
            wf.save_points(tests[loop_count],data,points,points_type,num_points)
            loop_count += 1

def run_linedisplay_c(sets,data,distance):
    os.system('cls')
    for i in range(0,len(sets),2):
        tests = np.linspace(int(sets[i]),int(sets[i+1]),int(sets[i+1]-sets[i]+1))
        heat_maps = wf.load_heatmaps(tests,data)
        coordinates = wf.get_line_coordinates(tests,data,distance)
        if heat_maps is 0 or coordinates is 0:
            return
        k = 0
        for j in heat_maps:
            usr = input('Continue (\'b\' to go back)')
            if usr == 'b':
                return
            wf.display_linedisplay(j,coordinates[k,:])
            k+=1
