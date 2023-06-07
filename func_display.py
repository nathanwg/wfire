from cProfile import label
from cgi import test
from heapq import nsmallest
from json import load
from tabnanny import check
from tkinter import E
from cv2 import threshold
import numpy as np
import os
from classes import Test
from func_wfipa import readfile, get_image_properties
import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv
import keyboard
import matplotlib.animation as animation
import func_wfipa
import matplotlib.patches as pat
from pynput import keyboard
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

def burnout_display(test,cmap_usr):
    usr = input('Continue (\'b\' to go back)')
    if usr == 'b':
        return 999
    fpath = os.getcwd() + '_cache\\burnout\\' + test.filename.replace('.tif','_burnoutframes.npy')
    ischeck = checkfile(fpath,test,checktype=False,isinput=True)
    if ischeck is False:
        return 999
    burnout_img = np.load(fpath)
    plt.imshow(burnout_img,cmap=cmap_usr)
    show_window(noticks=True,winmax=True,closewin=True,showwin=True)

def calc_uncertainty(arr,n):
    if n != 10:
        tval = input('What is the t-distribution value you are using?: ')
    else:
        tval = 2.262
    unc = tval*np.std(arr)/np.sqrt(n)
    return unc

def display_igloc(test,map_tags,cmap):
    maps = []
    for i in map_tags:
        map = load_heatmap(test,i)
        map = map/map.max()
        maps.append(map)
    img01,img02 = np.concatenate((maps[0],maps[1]),axis=1),np.concatenate((maps[2],maps[3]),axis=1)
    imgs = np.concatenate((img01,img02),axis=0)
    usr = displaymaps(imgs,'igloc',cmap)
    return usr

def display_linedisplay(h,coordinates):
    y_c = coordinates[0]
    x_L = coordinates[1]
    dx = coordinates[2]
    num_cols = h.shape[1]
    for i in range(0,num_cols):
        y = round(y_c)
        if y >= 256:
            y = 255
        h[y,i] = 4500
    plt.close()
    plt.imshow(h,cmap='nipy_spectral_r')
    show_window(noticks=True,winmax=False,closewin=True,showwin=True)

def display_mapsets_c(test,cmap_usr):
    usr = input('Continue or go back (b)')
    if usr == 'b':
        return 999
    cwd = os.getcwd()
    name = test.filename.replace('.tif','')
    item = 1
    maps = []
    loadpath = cwd+'_cache\\heatmaps\\cbsets\\' + name + '_cbsets03' + '.npy'
    map = np.load(loadpath)
    titles = ('0-0.1 -- 0.02','0-0.5 -- 0.1','0-1.0 -- 0.2')
    for i in range(4):
        loadpath = cwd+'_cache\\heatmaps\\cbsets\\' + name + '_cbsets0' + str(item) + '.npy'
        map = np.load(loadpath)
        # input(map)
        map = map/map.max()
        maps.append(map)
        # plt.imshow(map,cmap=cmap_usr)
        # plt.title(titles[i])
        # ax = plt.gca()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        # show_window(noticks=True,winmax=False,closewin=True)
        item+=1
    imgs01 = np.concatenate((maps[0],maps[1]),axis=1)
    imgs02 = np.concatenate((maps[2],maps[3]),axis=1)
    img = np.concatenate((imgs01,imgs02),axis=0)
    plt.figure()
    plt.imshow(img,cmap=cmap_usr)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    show_window(noticks=True,winmax=True,closewin=False,showwin=True)
    return True

def display_mapsets_d(test,cmap_usr):
    usr = input('Continue or go back (b)')
    if usr == 'b':
        return 999
    cwd = os.getcwd()
    name = test.filename.replace('.tif','')
    loadpaths = [cwd+'_cache\\heatmaps\\dsets\\' + name + '_dsets06' + '.npy',cwd+'_cache\\heatmaps\\dsets2\\' + name + '_dsets06' + '.npy']
    maps_load = [np.load(loadpaths[0]),np.load(loadpaths[1])]
    max_vals = [maps_load[0].max(),maps_load[1].max()]
    xscalars = [0.6/max_vals[0],2.52/max_vals[1]]
    num_rows = maps_load[0].shape[0]
    calibs = [np.zeros((num_rows,1)),np.zeros((num_rows,1))]
    calibs[0][0,0] = max_vals[0]
    calibs[1][0,0] = max_vals[1]

    folder_paths = ['_cache\\heatmaps\\dsets\\','_cache\\heatmaps\\dsets2\\']
    plots = []
    coors = [(200,50),(900,50),(200,600),(900,600)]
    for j in range(2):
        item = 1
        maps = []
        for i in range(7):
            loadpath = cwd+folder_paths[j] + name + '_dsets0' + str(item) + '.npy'
            map = np.load(loadpath)
            map = np.concatenate((map,calibs[j]),axis=1)
            map*=xscalars[j]
            maps.append(map)
            item+=1
        img01 = np.concatenate((maps[0],maps[1],maps[2]),axis=1)
        img02 = np.concatenate((maps[3],maps[4],maps[5]),axis=1)

        img = np.concatenate((img01,img02),axis=0)
        plots.append(plt.figure())
        move_figure(plots[j*2],coor=coors[j*2])
        plt.rcParams["figure.autolayout"] = True
        plt.imshow(img,cmap=cmap_usr)
        plt.colorbar()
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        show_window(noticks=True,winmax=False,closewin=False,showwin=False)

        plots.append(plt.figure())
        move_figure(plots[j*2+1],coor=coors[j*2+1])
        plt.imshow(maps[-1],cmap=cmap_usr)
        plt.colorbar()
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if j == 0:
            show_window(noticks=True,winmax=False,closewin=False,showwin=False)
        elif j == 1:
            show_window(noticks=True,winmax=False,closewin=False,showwin=True)
    plt.close('all')
    return True

def displayarea(test,cmap_usr):
    show = True
    if test.fmc == 0:
        threshold = 50
    else:
        threshold = 35
    frame_num = load_area(test)[1]
    if frame_num is None:
        return 999
    ffilepath = os.getcwd().replace('wfire','wfire_cache\\flame_area\\frames\\')+test.filename.replace('.tif','_areaframe_uncropped.npy')
    ischeck = checkfile(ffilepath,test,checktype=False,isinput=False)
    if ischeck == False:
        fname = os.getcwd().replace('wfire','') + test.filename
        img = readfile(fname,True)[0]
        frames = get_image_properties(img)[0]
        m_frame = frames[int(frame_num)]
        np.save(ffilepath,m_frame)
        show = False
    else:
        m_frame = np.load(ffilepath)
    mfilepath = os.getcwd()+'_cache\\flame_area\\frames\\'+test.filename.replace('.tif','_areaframe_cropped.npy')
    ischeck = checkfile(mfilepath,test,checktype=False,isinput=True)
    if ischeck == False:
        return 999
    m_frame__ = np.load(mfilepath)
    dis = np.concatenate((m_frame,m_frame__),axis=1)
    num_rows = dis.shape[0]
    calib = np.zeros((num_rows,1))
    calib[0,0] = 255
    dis = np.concatenate((dis,calib),axis=1)
    dis_bool = ((dis - threshold)>=0)
    dis_new = np.multiply(dis,dis_bool)
    imgshow = np.concatenate((dis,dis_new),axis=0)

    areaframe_ell_path = os.getcwd()+'_cache\\flame_area\\frames\\'+test.filename.replace('.tif','_areaframe_ell.npy')
    if os.path.exists(areaframe_ell_path):
        areaframe_ell = np.load(areaframe_ell_path)
        rows,cols = areaframe_ell.shape[0],areaframe_ell.shape[1]
        if rows != cols:
            areas = calc_area(areaframe_ell,None,threshold,test.spatial_calibration,'frame')
            if areas[0] >= areas[1]:
                start,stop = 0,int(cols/2)
            elif areas[1] > areas[0]:
                start,stop = int(cols/2),cols
            areaframe_ell = areaframe_ell[0:rows,start:stop]
        areaframe_ell = np.concatenate((areaframe_ell,calib),axis=1)
        frame_bool = ((areaframe_ell-threshold)>=0)
        img_add = np.multiply(areaframe_ell,frame_bool)
        img_new = np.concatenate((areaframe_ell,img_add),axis=0)
        imgshow = np.concatenate((imgshow,img_new),axis=1)
    if show:
        # os.system('cls')
        print('\n\n\n\n\n','--------------------------------------------------')
        usr = input('Press \'Enter\' to continue, enter \'q\' to stop: ')
        if usr == 'q':
            return 999
        plt.imshow(imgshow,cmap=cmap_usr)
        title = 'Test number: '+str(int(test.testnumber))
        plt.title(title)
        show_window(noticks=True,winmax=True,closewin=True,showwin=True)
    return

def displaymaps(heatmap,map_type,cmap_usr):
    """...
    """
    if map_type != 'igloc':
        usr = input('Continue (\'b\' to go back)')
        if usr == 'b':
            return 999
        if heatmap is None:
            input('Heatmap has not been loaded, most likely because there is no file saved for it (Hit \'Enter\')')
            return 999
    num_rows = heatmap.shape[0]
    calib = np.zeros((num_rows,1))
    calib[0,0] = 4500
    if map_type == 'all':
        imgs = heatmap
        # imgs = np.concatenate((heatmap,calib),axis=1)
    else:
        imgs = heatmap
    plt.figure()
    plt.imshow(imgs,cmap=cmap_usr)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    # if map_type == 'ig':
    #     plt.colorbar()
    show_window(noticks=True,winmax=True,closewin=True,showwin=True)
    return True

def get_plotinfo(sets,data):
    """ This function gets plot info and returns
        to various plot functions.
    """
    labels,temperatures,linestyle = [],[],[]
    labeldried,labellive_m,labellive_i=False,False,False
    for i in range(0,len(sets),2):
        start = int(sets[i])
        stop = int(sets[i+1])
        tests = np.linspace(start,stop,stop-start+1)
        test = data[int(tests[0])-1]
        temperature = int(test.set_type[3])
        temperatures.append(temperature)
        if test.fmc == 0:
            linestyle.append('ko')
            if labeldried == True:
                labels.append(None)
            elif labeldried == False:
                labels.append('Oven-dried fuel-'+test.set_type[0])
                labeldried = True
        elif test.fmc != 0 and test.set_type[0] == 'groups':
            linestyle.append('g^')
            if labellive_m == True:
                labels.append(None)
            elif labellive_m == False:
                labels.append(test.set_type[0])
                labellive_m = True
        elif test.fmc != 0 and test.set_type[0] == 'individual':
            linestyle.append('bo')
            if labellive_i == True:
                labels.append(None)
            elif labellive_i == False:
                labels.append(test.set_type[0])
                labellive_i = True
    return labels,temperatures,linestyle

def move_figure(f,coor):
    """Move figure's upper left corner to pixel (x, y)"""
    x,y = coor[0],coor[1]
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

def plot_numpixelsarea(test,showmax):
    usr = input('Continue? (b)')
    if usr == 'b':
        return 999
    numpixels_filepath = os.getcwd() + '_cache\\numpixels\\' + test.filename.replace('.tif','_numpixels.npy')
    areavals_numpixels_filepath = os.getcwd() + '_cache\\flame_area\\vals_numpixels\\' + test.filename.replace('.tif','_areavals_numpixels.npy')
    msg = 'You need to run \'Check frame number\' to generate this file'
    ischeck = checkfile(numpixels_filepath,test,checktype=False,isinput=True)
    if ischeck == False:
        return None
    numpixels = np.load(numpixels_filepath)
    num_frames = len(numpixels)
    x,xlabel = np.linspace(1,num_frames,num_frames),'frame number'
    if showmax == False:
        x/=500
        xlabel = 'time (s)'

    pixel_length = test.spatial_calibration*100 # change from m to cm
    pixel_area = pixel_length**2
    areapixels = numpixels*pixel_area # cm^2

    plt.figure(figsize=[10,7],dpi=140)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 24})
    plt.tight_layout()
    plt.plot(x,areapixels,linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel('combustion area (cm$^2$)')
    title = 'Test number: '+str(test.testnumber)+'   file: '+test.filename
    # plt.title(title)
    if showmax:
        frame_num_cropped = load_area(test)[1]
        frame_num_numpixels = np.load(areavals_numpixels_filepath)
        x_c,y = [frame_num_cropped,frame_num_cropped],[0,np.amax(areapixels)/2]
        x_n = [frame_num_numpixels,frame_num_numpixels]
        plt.plot(x_c,y)
        plt.plot(x_n,y)
    show_window(noticks=False,winmax=False,closewin=True,showwin=True)
    return

def on_press(key):
    try:
        print()
    except AttributeError:
        print()

def on_release(key):
    print()
    if key == keyboard.Key.enter:
        # Stop listener
        return False

def plot_dur(sets,data,dur,showunc):
    labels,temperatures,linestyle = get_plotinfo(sets,data) 
    dur_averages = []
    plt.figure(figsize=[9,7],dpi=140)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 24})
    plt.tight_layout()
    for i in range(len(dur)):        
        dur_averages.append(np.mean(dur[i]))
        unc,cap = calc_uncertainty(dur[i],10),4
        if showunc == False:
            unc,cap = 0,0
        plt.errorbar(temperatures[i],dur_averages[i]/500,fmt=linestyle[i],yerr=unc/500,capsize=cap,label=labels[i])
        print('Temp: ',temperatures[i],'\nDuration: ',dur_averages[i]/500)
    input()
    plt.xlabel('Average exhaust gas temperature ($^{\circ}$C)')
    plt.ylabel('Average flaming duration (s)')
    # plt.title('Average flaming duration')
    plt.legend()
    show_window(noticks=False,winmax=False,closewin=True)
    return

def plot_igtime(sets,data,igtimes,showunc):
    """ Plots ignition times for different sets of tests
    """
    labels,temperatures,linestyle = get_plotinfo(sets,data) 
    igtimes_averages = []
    plt.figure(figsize=[9,7.5],dpi=140)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 24})
    plt.tight_layout()
    for i in range(len(igtimes)):        
        igtimes_averages.append(np.mean(igtimes[i]))
        unc,cap = calc_uncertainty(igtimes[i],10),4
        if showunc == False:
            unc,cap = 0,0
        plt.errorbar(temperatures[i],igtimes_averages[i]/500,fmt=linestyle[i],yerr=unc/500,capsize=cap,label=labels[i])
        print('\n',sets[2*i],sets[2*i+1])
        print('Temperature: ',temperatures[i],'igtime: ',igtimes_averages[i]/500)
    plt.xlabel('Average exhaust gas temperature ($^{\circ}$C)')
    plt.ylabel('Average ignition time (s)')
    # plt.title('Average ignition times')
    plt.legend()
    show_window(noticks=False,winmax=False,closewin=True,showwin=True)
    return

def plot_ima(sets,data,ima_sets,showunc):
    labels,temperatures,linestyle = get_plotinfo(sets,data) 
    ima_averages = []
    plt.figure(figsize=[9,7.5],dpi=140)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 24})
    plt.tight_layout()
    for i in range(len(ima_sets)):        
        ima_averages.append(np.mean(ima_sets[i]))
        unc,cap = calc_uncertainty(ima_sets[i],10),4
        if showunc == False:
            unc,cap = 0,0
        plt.errorbar(temperatures[i],ima_averages[i]/255,fmt=linestyle[i],yerr=unc/255,capsize=cap,label=labels[i])
    plt.xlabel('Average exhaust gas temperature $^{\circ}$C')
    plt.ylabel('Average intensity of max flame area')
    # plt.title('Average normalized light intensity of maximum flame area')
    plt.ylim(0,1)
    plt.legend()
    show_window(noticks=False,winmax=False,closewin=True)
    return

def plot_max_flame_area(sets,data,max_flamearea_sets,showunc):  
    labels,temperatures,linestyle = get_plotinfo(sets,data) 
    area_averages = []
    plt.figure(figsize=[9,7.5],dpi=140)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 24})
    plt.tight_layout()
    for i in range(len(max_flamearea_sets)):           
        area_averages.append(np.mean(max_flamearea_sets[i]))
        unc,cap = calc_uncertainty(max_flamearea_sets[i],10),4
        if showunc == False:
            unc,cap = 0,0
        plt.errorbar(temperatures[i],area_averages[i],fmt=linestyle[i],yerr=unc,capsize=cap,label=labels[i])
    plt.xlabel('Average exhaust gas temperature $^{\circ}$C')
    plt.ylabel('Maximum flame area (cm$^2$)')
    # plt.title('Average maximum flame area')
    plt.legend()
    show_window(noticks=False,winmax=False,closewin=True)
    return

def plotmedians(sets,data,medians_sets,showunc):
    labels,temperatures,linestyle = get_plotinfo(sets,data)   
    median_averages = []
    for i in range(len(medians_sets)):           
        median_averages.append(np.mean(medians_sets[i]))
        unc,cap = calc_uncertainty(medians_sets[i],10),4
        if showunc == False:
            unc,cap = 0,0
        plt.errorbar(temperatures[i],median_averages[i],fmt=linestyle[i],yerr=unc,capsize=cap,label=labels[i])
    plt.xlabel('Average exhaust gas temperature $^{\circ}C$')
    plt.ylabel('Median of flame activity')
    plt.title('Average median of flame detection during burning')
    plt.legend()
    plt.ylim(0,1)
    show_window(noticks=False,winmax=False,closewin=True)
    return

def plotprofiles_h(sets,data,distance,isnorm,ylim):
    labels = []
    for i in range(0,len(sets),2):
        start = int(sets[i])
        stop = int(sets[i+1])
        tests = np.linspace(start,stop,stop-start+1)
        temperature = int(data[int(tests[0])-1].set_type[3])
        temp_label = 'avg T = '+str(temperature)+' C'
        labels.append(temp_label)
        
        line_avg = 0
        line_avg_norm = 0
        labels = ['avg T = 460','avg T = 520','avg T = 610','avg T = 670','avg T = 880']
        num = len(tests)
        for ii in range(0,num):
            test = data[int(tests[ii])-1]
            heatmap = load_heatmap(test)
            if heatmap is None:
                return
            num_cols = heatmap.shape[1]
            line = np.zeros((1,num_cols))
            coordinates = get_line_coordinates(test,distance)
            if coordinates is 0:
                return
            y_c,x_L,dx = coordinates[0,0],coordinates[0,1],coordinates[0,2]
            check = False
            for j in range(0,num_cols):
                y = round(y_c)
                if y>=256:
                    y = 255
                if y<0 and check is False:
                    print('error: y = ',y,'\n','This is set: ',start,'-',stop,'\n And test number: ',ii+1,'\n')
                    usr = input('Continue? (y/n)')
                    if usr == 'y':
                        check = True
                    elif usr == 'n':
                        return
                line[0,j] = heatmap[y,j]

            eof = test.eof
            line_norm = line/(eof-test.ignition_time[1])
            line_avg += line
            line_avg_norm += line_norm
        line_avg /= num
        line_avg_norm /= num
        x_plot = np.linspace(0,9,num_cols)
        if isnorm is True:
            plt.plot(x_plot,line_avg_norm.T,label=labels[int(i/2)])
            axlabel = 's/s'
        elif isnorm is False:
            plt.plot(x_plot,line_avg.T/500,label=labels[int(i/2)])
            axlabel = 't (s)'
    print(distance[0],' cm')
    plt.xlabel('x (cm)')
    plt.ylabel(axlabel)
    plt.ylim(0,ylim)
    plt.xlim(0,9)
    plt.legend()
    show_window(noticks=False,winmax=False,closewin=True)

def plotprofiles_v(sets,data,isnorm,xlim):
    labels = []
    for i in range(0,len(sets),2):
        start = int(sets[i])
        stop = int(sets[i+1])
        tests = np.linspace(start,stop,stop-start+1)
        temperature = int(data[int(tests[0])-1].set_type[3])
        temp_label = 'avg T = '+str(temperature)+' C'
        labels.append(temp_label)

        line_avg = 0
        line_avg_norm = 0
        num = len(tests)
        for ii in range(0,num):
            test = data[int(tests[ii])-1]
            heatmap = load_heatmap(test)
            if heatmap is None:
                return
            num_rows = heatmap.shape[0]
            pathname = test.filename.replace('.tif','')
            cwd = os.getcwd()
            path = cwd+'\\points_vert\\'+pathname+'_points_vert.txt'
            ischeck = checkfile(path,test,checktype=False,isinput=True)
            if ischeck == False:
                return
            p = np.loadtxt(path,unpack=True)
            x = round(p[0])
            line = np.zeros((1,num_rows))
            check = False
            for j in range(0,num_rows):
                line[0,-1*j+num_rows-1] = heatmap[j,x]
            eof = test.eof
            line_norm = line/(eof-test.ignition_time[1])
            line_avg += line
            line_avg_norm += line_norm
        line_avg /= num
        line_avg_norm /= num
        y_plot = np.linspace(0,9,num_rows)
        if isnorm is True:
            plt.plot(line_avg_norm.T,y_plot,label=labels[int(i/2)])
            axlabel = 's/s'
        elif isnorm is False:
            plt.plot(line_avg.T/500,y_plot,label=labels[int(i/2)])
            axlabel = 't (s)'
    plt.ylabel('y (cm)')
    plt.xlabel(axlabel)
    plt.xlim(0,xlim)
    plt.ylim(0,9)
    plt.legend()
    show_window(noticks=False,winmax=False,closewin=True)

def show_frames(frames,foi,eof):
    """
    """
    sec_scale = 0.5
    last_frame = foi+2500*sec_scale
    if last_frame > len(frames):
        input('Error')
        return
        
    imgs01 = np.concatenate((frames[foi],frames[int(foi+500*sec_scale)],frames[int(foi+1000*sec_scale)]),axis=1)
    imgs02 = np.concatenate((frames[int(foi+1500*sec_scale)],frames[int(foi+2000*sec_scale)],frames[int(foi+2500*sec_scale)]),axis=1)
    scale = 1.75
    width = int(imgs01.shape[1]*scale)
    height = int(imgs02.shape[0]*scale)
    dim = (width,height)
    imgs01 = cv.resize(imgs01,dim)
    imgs02 = cv.resize(imgs02,dim)
    imgs = np.concatenate((imgs01,imgs02),axis=0)
    maps = ['viridis','twilight','turbo','CMRmap','flag','gist_ncar','nipy_spectral','tab20','Set3']
    maps = ['turbo']
    colorline=['white','black']
    time = []
    for i in range(6):
        t = str(round((foi+500*sec_scale*i)/500,2))+ ' s (+' + str(round(i*sec_scale,2)) + ' s)'
        time.append(t)
    x = [10,460,905]
    y = [440,880]
    xline = [[0,1343],[447,447],[895,895]]
    yline = [[448,448],[0,895],[0,895]]
    for i in maps:
        plt.imshow(imgs,cmap=i)
        plt.title(i)
        c = 0
        for j in y:
            for k in x:
                plt.text(k,j,time[c],size=14,color=colorline[maps.index(i)])
                c+=1
        for ii in range(3):
            plt.plot(xline[ii],yline[ii],color=colorline[maps.index(i)],linewidth=0.5)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        show_window(noticks=True,winmax=True,closewin=True)

    ignition_frame = int(foi)
    img01 = frames[ignition_frame]
    img02 = frames[int(ignition_frame+500*sec_scale)]
    img03 = frames[int(ignition_frame+1000*sec_scale)]
    img04 = frames[int(ignition_frame+1500*sec_scale)]
    img05 = frames[int(ignition_frame+2000*sec_scale)]
    img06 = frames[int(ignition_frame+2500*sec_scale)]
    imglist = [img01,img02,img03,img04,img05,img06]
    font = cv.FONT_HERSHEY_SIMPLEX
    bottomLeft = (10,20)
    fontScale = 0.5
    fontColor = (255,255,255)
    thickness = 1
    lineType = 3
    for i in range(6):
        cv.putText(imglist[i],time[i],bottomLeft,font,fontScale,fontColor,thickness,lineType)
    imgs01 = np.concatenate((img01,img02,img03),axis=1)
    imgs02 = np.concatenate((img04,img05,img06),axis=1)
    imgs01 = cv.resize(imgs01,dim)
    imgs02 = cv.resize(imgs02,dim)
    img = np.concatenate((imgs01,imgs02),axis=0)
    cv.imshow("window",img)
    while cv.getWindowProperty('window', cv.WND_PROP_VISIBLE) > 0:
        k = cv.waitKey(50)
        if k != -1:
            cv.destroyAllWindows()
            break

def show_ignition(test):
    """.....
    """
    file = os.getcwd().replace('wfire','') + test.filename
    img = readfile(file,True)[0]
    frames = img[1]
    show_frames(frames,test.ignition_time[1],test.eof)

def show_window(noticks,winmax,closewin,showwin):
    if noticks:
        plt.tick_params(axis='both',bottom=False,labelbottom=False,left=False,labelleft=False)
    if winmax:
        plt.get_current_fig_manager().window.showMaximized()
    if showwin:
        plt.show(block=False)
        run = True
        plt.pause(0.5)
        listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
        listener.start()
        while run:
            plt.waitforbuttonpress(timeout=0.1)
            isfig = bool(plt.get_fignums())
            if isfig == False or listener.running == False:
                run = False
    if closewin is True:
        plt.close('all')
    return

