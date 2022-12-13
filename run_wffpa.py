from asyncore import loop
from cgi import test
from cmath import nan
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
            loop_handl(sets,data,'linedisplay',args=[distance])
        elif usr == 's':
            print('Distance values: ',distance)
            input(' ')
        elif usr == 'b':
            running = False
    return

def run_changeparameters(distance,sets,filename,ylim,cmap,cmap_filepath,showunc):
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------','\n')
    print(' Change test numbers ---- n')
    print(' Change line parameters - p')
    print(' Change plot ylimit ----- y')
    print(' Change cmap ------------ c')
    print(' Error bars ------------- e')
    print(' Go back ---------------- b')
    usr = input('Selected option: ')
    if usr == 'n':
        sets = wf.change_tests()
    elif usr == 'p':
        distance = wf.change_linepar(distance)
    elif usr == 'y':
        ylim = wf.change_ylim(ylim)
    elif usr == 'c':
        cmap = wf.change_cmap(cmap)
    elif usr == 'e':
        showunc = wf.change_errbar(showunc)
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
    file.write(' ')
    file.write(str(showunc))
    file.close()
    np.save(cmap_filepath,cmap)

    return distance,sets,ylim,cmap,bool(showunc)

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
    loop_handl(sets,data,'showig',None)
    return

def run_flametimeline(sets,data):
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------','\n')
    print(' This will generate and save plots for the following test numbers:')
    for i in range(0,len(sets),2):
        print(round(sets[i]),'-',round(sets[i+1]))
    usr = input('Would you like to continue? (y/n)')
    if usr == 'y':
        loop_handl(sets,data,'timeline',[])
    else:
        return
    return

def run_createplots(sets,data,distance,ylim,showunc):
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------','\n')
    print('Type option from the following list and hit enter:')
    print(' Plot flame profiles ------------- p')
    print(' Plot flame timelines ------------ t')
    print(' Plot ignition times ------------- i')
    print(' Plot avgerage light intensity --- a')
    print(' Plot timeline medians ----------- m')
    print(' Plot max flame areas ------------ r')
    print(' Plot avg int of max flame areas - s')
    usr = input('Selected option: ')
    if usr == 'p':
        run_plotprofiles(sets,data,distance,ylim)
    elif usr == 't':
        run_flametimeline(sets,data)
    elif usr == 'i':
        run_plotigtime(sets,data,showunc)
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
        run_plotmedians(sets,data,showunc)
    elif usr == 'r':
        run_plot_max_flame_area(sets,data,showunc)
    elif usr == 's':
        run_plot_ima(sets,data,showunc)
    else:
        input('Error (hit \'Enter\' to continue)')
        return

def run_plotigtime(sets,data,showunc):
    igtimes = loop_handl(sets,data,'igtimes')
    wf.plot_igtime(sets,data,igtimes,showunc)

def run_plotmedians(sets,data,showunc):
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
    wf.plotmedians(sets,data,medians_sets,showunc)

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

def run_validate(sets,data,distance,cmap):
    running,error = True, False
    while running is True:
        os.system('cls')
        print('\n\n\n\n\n','--------------------------------------------------','\n')
        print('Type option from the following list and hit enter:')
        print(' Show line position -- n')
        print(' Show ignition ------- i')
        print(' Check grids --------- g')
        print(' Area validation ----- a')
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
            loop_handl(sets,data,'grid',None)
        elif usr_func == 'a':
            run_validatearea(sets,data,cmap)
        elif usr_func == 'b':
            return
        else:
            error = True
    return

def run_validatearea(sets,data,cmap):
    running,error = True,False
    while running is True:
        os.system('cls')
        print('\n\n\n\n\n','--------------------------------------------------','\n')
        print('Type option from the following list and hit enter:')
        print(' Display max area -------------- m')
        print(' Check frame number ------------ n')
        print(' Calculate percent saturated --- s')
        print(' Check numpixel plot ----------- p')
        print(' Go back ----------------------- b')
        for i in range(0,len(sets),2):
            print(round(sets[i]),'-',round(sets[i+1]))
        print()
        if error is True:
            print('Error -- selected invalid option')
            error = False
            print()
        usr = input('Selected option: ')
        if usr == 'm':
            loop_handl(sets,data,'displayarea',args=[cmap])
        elif usr == 'n':
            loop_handl(sets,data,'checkframenum',args=[cmap])
        elif usr == 's':
            loop_handl(sets,data,'satpercent',None)
        elif usr == 'p':
            loop_handl(sets,data,'numpixelsarea',None)
        elif usr == 'b':
            running = False
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
            loop_handl(sets,data,'area',[])
        elif usr_func == 'i':
            loop_handl(sets,data,'ima',[])
        elif usr_func == 'b':
            return
    return

def run_plot_max_flame_area(sets,data,showunc):
    max_flame_area_sets = loop_handl(sets,data,'pltarea',None)
    wf.plot_max_flame_area(sets,data,max_flame_area_sets,showunc)

def run_plot_ima(sets,data,showunc):
    ima_sets = loop_handl(sets,data,'pltima',None)
    wf.plot_ima(sets,data,ima_sets,showunc)

def loop_handl(sets,data,tag,args):
    nsets = len(sets)
    outputs = []
    for i in range(0,nsets,2):
        start = int(sets[i])
        stop = int(sets[i+1])
        tests = np.linspace(start,stop,stop-start+1)
        ntests = len(tests)

        switch_out = []
        for j in range(ntests):
            num = int(tests[j])-1
            test = data[num]
            switch_out.append(func_switch(test,tag,args))
            if switch_out[j] == 999:
                return outputs
        outputs.append(switch_out)
    return outputs

def func_switch(test,tag,args):
    func_out = None
    if tag == 'ima':
        func_out = wf.get_ima(test)
    elif tag == 'area':
        func_out = wf.get_max_flame_area(test)
    elif tag == 'grid':
        func_out = wf.creategrids(test)
    elif tag == 'avgint':
        wf.calc_avgint(test,args)
    elif tag == 'timeline':
        wf.get_flametimeline(test)
    elif tag == 'pltarea':
        func_out = wf.load_area(test)[0]
    elif tag == 'pltima':
        func_out = wf.get_ima(test)
    elif tag == 'selectpoints':
        heatmap = wf.load_heatmap(test)
        points,num_points = wf.get_points(heatmap,test,points_type=args[0])
        wf.save_points(test,points,num_points,points_type=args[0])
    elif tag == 'linedisplay':
        func_out = rn.run_linedisplay_c(test,distance=args[0])
    elif tag == 'igtimes':
        func_out = test.ignition_time[1]
    elif tag == 'getmap':
        func_out = wf.get_heatmaps(test,save=args[1],thresh=args[0])
    elif tag == 'displaymap':
        heatmap = wf.load_heatmap(test)
        func_out = wf.displaymaps(heatmap)
    elif tag == 'showig':
        wf.show_ignition(test)
    elif tag == 'displayarea':
        func_out = wf.displayarea(test,cmap_usr=args[0])
    elif tag == 'checkframenum':
        func_out = wf.checkframenum(test,cmap_usr=args[0])
    elif tag == 'satpercent':
        func_out = wf.calc_saturate(test)
    elif tag == 'numpixelsarea':
        func_out = wf.plot_numpixelsarea(test)
    else:
        return func_out
    return func_out
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
def main():
    data = wf.importdata()
    filename = 'cache_wffpa.txt'
    cache = np.loadtxt(filename,unpack=True)
    sets = []
    for i in range(0,len(cache)-4):
        sets.append(cache[i])
    distance = [cache[-5],cache[-4],cache[-3]]
    ylim = float(cache[-2])
    showunc = bool(cache[-1])
    cmap_filepath = os.getcwd() + '_cache\\cmap.npy'
    cmap = str(np.load(cmap_filepath))
    running,error = True, False
    while running is True:
        os.system('cls')
        print('\n\n\n\n\n','--------------------------------------------------')
        print('   ------------------Main Menu-------------------')
        print(' --------------------------------------------------','\n')
        print('Type option from the following list and hit enter:','\n')
        print('  Data Analysis Tools ------------ T')
        print('  Create Analysis Plots ---------- P')
        print('  Data Analysis Validation ------- V')
        print('  Settings ----------------------- S')
        print('  Quit Program ------------------- Q','\n')
        print(' The following test numbers are being considered:')
        sets_list = ''
        for i in range(0,len(sets),2):
            print(round(sets[i]),'-',round(sets[i+1]))
        print()
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
            run_createplots(sets,data,distance,ylim,showunc)
        elif usr_func == 's' or usr_func == 'S':
            distance,sets,ylim,cmap,showunc = run_changeparameters(distance,sets,filename,ylim,cmap,cmap_filepath,showunc)
            ylim = float(ylim)
        elif usr_func == 'v' or usr_func == 'V':
            run_validate(sets,data,distance,cmap)
        elif usr_func == 'q' or usr_func == 'Q':
            running = False
        else:
            error = True
    return
            
            
            
            
