from cgi import test
from func_wfipa import readfile, get_image_properties
import func_wffpa as wf
import func_run as rn
import numpy as np
import os

def run_heatmap(sets,data):
    running = True
    heat_maps = None
    while running is True:
        os.system('cls')
        print('\n\n\n\n\n','--------------------------------------------------','\n')
        print(' Generate heatmap ---- g','\n','Display heatmap ----- d')
        print(' Go back ------------- b')
        usr = input('Selected option: ')
        if usr == 'g':
            heat_maps = rn.run_heatmap_g(sets,data)
        elif usr == 'd':
            rn.run_heatmap_d(sets,heat_maps,data)
        elif usr == 'b':
            running = False
    return heat_maps

def run_selectpoints(sets,data):
    running = True
    while running is True:
        os.system('cls')
        print('\n\n\n\n\n','--------------------------------------------------','\n')
        print(' Select points ------- s')
        print(' Go back ------------- b')
        usr = input('Selected option: ')
        if usr == 's':
            rn.run_selectpoints_s(sets,data)
        elif usr == 'b':
            running = False
    return

def run_linedisplay(sets,data,distance):
    running = True
    while running is True:
        os.system('cls')
        print('\n\n\n\n\n','--------------------------------------------------','\n')
        print(' Confirm linedisplay -- c')
        print(' Show distance values - s')
        print(' Go back -------------- b')
        usr = input('Selected option: ')
        if usr == 'c':
            rn.run_linedisplay_c(sets,data,distance)
        elif usr == 's':
            print('Distance values: ',distance)
            input(' ')
        elif usr == 'b':
            running = False
    return

def run_changeparameters(distance,sets,filename,ylim):
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------','\n')
    print(' Change test numbers ---- n')
    print(' Change line parameters - p')
    print(' Change plot ylimit ----- y')
    print(' Go back ---------------- b')
    usr = input('Selected option: ')
    if usr == 'n':
        sets = wf.change_tests()
    elif usr == 'p':
        distance = wf.change_linepar(distance)
    elif usr == 'y':
        ylim = wf.change_ylim(ylim)
    elif usr == 'b':
        return distance,sets,ylim
    else:
        print()
    file = open(filename,'w')
    for i in range(int(len(sets))):
        if i != 0:
            file.write(' ')
        file.write(str(sets[i]))
    for i in range(0,3):
        file.write(' ')
        file.write(str(distance[i]))
    file.write(' ')
    file.write(str(ylim))
    file.close()
    return distance,sets,ylim

def run_plotprofiles(sets,data,distance,ylim):
    os.system('cls')
    isnorm = True
    print('\n\n\n\n\n','--------------------------------------------------','\n')
    usr = input('Are you plotting a horizontal or vertical profile? (h/v)')
    if usr == 'h':    
        usr = input('Regular or normalized plot? (r/n)')
        if usr == 'r':
            isnorm = False
        wf.plotprofiles_h(sets,data,distance,isnorm,ylim)
    elif usr == 'v':
        usr = input('Regular or normalized plot? (r/n)')
        if usr == 'r':
            isnorm = False
        wf.plotprofiles_v(sets,data,isnorm,ylim)
    return

def run_showignition(sets,data):
    for i in range(0,len(sets),2):
        tests = np.linspace(int(sets[i]),int(sets[i+1]),int(sets[i+1]-sets[i]+1))
        run = wf.show_ignition(tests,data)
        if run is False:
            break
    return

def run_flametimeline(sets,data):
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------','\n')
    print(' This will generate and save plots for the following test numbers:')
    for i in range(0,len(sets),2):
        print(round(sets[i]),'-',round(sets[i+1]))
    usr = input('Would you like to continue? (y/n)')
    if usr == 'y':
        wf.get_flametimeline(sets,data)
    else:
        return
    return

def run_createplots(sets,data,distance,ylim):
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------','\n')
    print('Type option from the following list and hit enter:')
    print(' Plot flame profiles ------------ p')
    print(' Plot flame timelines ----------- t')
    print(' Plot ignition times ------------ i')
    print(' Plot avgerage light intensity -- a')
    print(' Plot timeline medians ---------- m')
    usr = input('Selected option: ')
    if usr == 'p':
        run_plotprofiles(sets,data,distance,ylim)
    elif usr == 't':
        run_flametimeline(sets,data)
    elif usr == 'i':
        run_plotigtime(sets,data)
    elif usr == 'a':
        ans = input(' Is the fuel evaluated in these tests live or oven dried? (L/O/b)')
        if ans == 'l':
            threshold = 35
        elif ans == 'o':
            threshold = 50
        else:
            return
        wf.calc_avgint(sets,data,threshold)
    elif usr == 'm':
        run_plotmedians(sets,data)
    else:
        input('Error (hit \'Enter\' to continue)')
        return

def run_plotigtime(sets,data):
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------','\n')
    print('The following settings for the plot are: ')
    filename = 'cache_plot_settings.txt'
    settings = np.loadtxt(filename,unpack=True)
    set_type = [settings[0],settings[1]]
    for i in set_type:
        if i == 1:
            i = 'Individual needle on stem'
        elif i == 2:
            i = 'Multiple needles'
        elif i == 0:
            i = 'Individual needle'
    set_condition = [settings[-2],settings[-1]]
    for i in set_condition:
        if i == 1:
            i = 'Live'
        elif i == 0:
            i = 'Oven-dried'
    set_height = settings[-2]
    temps = []
    temps_num = len(settings)-3
    for i in range(temps_num):
        temps.append(settings[i+1])
    print(' Set type:',set_type)
    print(' Temperatures:',temps)
    print(' Height:',set_height)
    print(' Condition:',set_condition)
    input()
    return

def run_plotmedians(sets,data):
    medians_sets = []
    for i in range(0,len(sets),2):
        start = int(sets[i])
        stop = int(sets[i+1])
        tests = np.linspace(start,stop,stop-start+1)

        medians = np.array(())
        for j in range(0,len(tests)):
            test = data[int(tests[j])-1]
            median_test = wf.get_median(test)
            medians = np.append(medians,median_test)
        medians_sets.append(medians)
    wf.plotmedians(sets,data,medians_sets)

def run_tools(sets,data):
    running,error = True, False
    while running is True:
        os.system('cls')
        print('\n\n\n\n\n','--------------------------------------------------','\n')
        print('Type option from the following list and hit enter:')
        print(' Heatmaps ------------ h','\n','Select points ------- s')
        print(' Flame area ---------- a')
        print(' Go back ------------- b')
        print(' The following test numbers are being considered:')
        for i in range(0,len(sets),2):
            print(round(sets[i]),'-',round(sets[i+1]))
        print()
        if error is True:
            print('Error -- selected invalid option')
            error = False
            print()
        usr_func = input('Selected option: ')
        if usr_func == 'h':
            heat_maps = run_heatmap(sets,data)
        elif usr_func == 's':
            run_selectpoints(sets,data)
        elif usr_func == 'a':
            run_flamearea(sets,data)
        elif usr_func == 'b':
            return
        else:
            error = True
    return

def run_validate(sets,data,distance):
    running,error = True, False
    while running is True:
        os.system('cls')
        print('\n\n\n\n\n','--------------------------------------------------','\n')
        print('Type option from the following list and hit enter:')
        print(' Show line position -- n')
        print(' Show ignition ------- i')
        print(' Check grids --------- g')
        print(' Go back ------------- b')
        print(' The following test numbers are being considered:')
        for i in range(0,len(sets),2):
            print(round(sets[i]),'-',round(sets[i+1]))
        print()
        if error is True:
            print('Error -- selected invalid option')
            error = False
            print()
        usr_func = input('Selected option: ')
        if usr_func == 'n':
            run_linedisplay(sets,data,distance)
        elif usr_func == 'i':
            run_showignition(sets,data)
        elif usr_func == 'g':
            wf.creategrids(sets,data)
        elif usr_func == 'b':
            return
        else:
            error = True
    return

def run_flamearea(sets,data):
    running,error = True, False
    while running is True:
        os.system('cls')
        print('\n\n\n\n\n','--------------------------------------------------','\n')
        print('Type option from the following list and hit enter:')
        print(' Get max flame area ---------------------- a')
        print(' Get intensity data of max flame area ---- i')
        print(' Go back --------------------------------- b')
        usr_func = input('Selected option: ')
        if usr_func == 'a':
            wf.get_max_flame_area(sets,data)
        elif usr_func == 'i':
            wf.get_ima(sets,data)
        elif usr_func == 'b':
            return
    return
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
def main():
    data = wf.importdata()
    filename = 'cache_wffpa.txt'
    cache = np.loadtxt(filename,unpack=True)
    sets = []
    for i in range(0,len(cache)-4):
        sets.append(cache[i])
    distance = [cache[-4],cache[-3],cache[-2]]
    ylim = float(cache[-1])
    running,error = True, False
    while running is True:
        os.system('cls')
        print('\n\n\n\n\n','--------------------------------------------------')
        print('   ------------------Main Menu-------------------')
        print(' --------------------------------------------------','\n')
        print('Type option from the following list and hit enter:','\n')
        print('  Data Analysis Tools ------------ T')
        print('  Create Analysis Plots ---------- P')
        print('  Change Analysis Parameters ----- C')
        print('  Data Analysis Validation ------- V')
        print('  Quit Program ------------------- Q','\n')
        print(' The following test numbers are being considered:')
        sets_list = ''
        for i in range(0,len(sets),2):
            if i == len(sets)-2:
                sets_list = sets_list + str(round(sets[i]))+'-'+str(round(sets[i+1]))
            else:    
                sets_list = sets_list + str(round(sets[i]))+'-'+str(round(sets[i+1]))+', '
        print('\n',sets_list,'\n')
        # print()
        # print(' Line currently being evaluated at ',distance[0],' cm\n')
        # print(' ylim for plot is set at: ',ylim,'\n')
        if error is True:
            print('Error -- selected invalid option')
            error = False
            print()
        usr_func = input(' Selected option: ')
        if usr_func == 't' or usr_func == 'T':
            run_tools(sets,data)
        elif usr_func == 'p' or usr_func == 'P':
            run_createplots(sets,data,distance,ylim)
        elif usr_func == 'c' or usr_func == 'C':
            distance,sets,ylim = run_changeparameters(distance,sets,filename,ylim)
            ylim = float(ylim)
        elif usr_func == 'v' or usr_func == 'V':
            run_validate(sets,data,distance)
        elif usr_func == 'q' or usr_func == 'Q':
            running = False
        else:
            error = True
    return
            
            
            
            
