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
from classes import Line
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
import mplcursors as mpl

def importdata(datafile='data.txt',namesfile='filenames.txt'):
    """
    Loads data from txt files, uses data to create a list of Test objects
    and returns this list as a tuple
    """
    month,day,year,testnum,set_,orientation,height,temp,fmc,time,frame,spatial,eof,flame_height = np.loadtxt(datafile,unpack=True)
    filenames = np.loadtxt(namesfile,dtype=str)
    data = []
    # print(len(filenames),len(set_))
    for i in range(0,len(filenames)):
        # print(i)
        filename = filenames[i]+'.tif'
        if set_[i] == 0 or set_[i] == 1:
            stype = 'individual'
        elif set_[i] == 2:
            stype = 'groups'
        else:
            stype = None
        ig_test = Test(filename,(month[i],day[i],year[i]),testnum[i],(stype,orientation[i],height[i],temp[i]),fmc[i],(time[i],int(frame[i])),spatial[i],eof[i],flame_height[i])
        data.append(ig_test)
    return tuple(data)

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

def get_quartermap(frames,num_rows,num_cols,thresh):
    heatmap = np.zeros((num_rows,num_cols))
    for j in frames:
        j = j.astype(float)
        arr = (j-thresh)>0
        heatmap+=arr
    return heatmap

def get_heatmaps(test,save,thresh,map_type):
    """.....
    """
    print(test.testnumber)
    name = test.filename.replace('.tif','')
    cwd = os.getcwd()
    if map_type == 'all':
        savepath = cwd+'\\heatmaps\\'+name+'_heatmap.npy'
    elif map_type == 'preig':
        savepath = cwd+'_cache\\heatmaps\\preig\\' + test.filename.replace('.tif','_preig_heatmap.npy')
    elif map_type == 'ig':
        savepath = cwd+'_cache\\heatmaps\\ig\\' + test.filename.replace('.tif','_ig_heatmap.npy')
    elif map_type == 'dis_ig':
        savepath = cwd+'_cache\\heatmaps\\dis_ig\\' + test.filename.replace('.tif','_disig_heatmap.npy')
    elif map_type == 'dis_c':
        savepath = cwd+'_cache\\heatmaps\\dis_c\\' + test.filename.replace('.tif','_disc_heatmap.npy')
    elif map_type == 'flaming':
        savepath = cwd+'_cache\\heatmaps\\flaming\\'+test.filename.replace('.tif','_flaming_heatmap.npy')
    ischeck = checkfile(savepath,test,checktype=True,isinput=True)
    if ischeck == False:
        return
    filepath = os.getcwd().replace('wfire','') + test.filename
    img,filename = readfile(filepath,True)
    frames,num_frames,num_rows,num_cols = get_image_properties(img)
    heatmap = np.zeros((num_rows,num_cols))
    frame_num = 0
    ignition_frame = test.ignition_time[1]
    fps = 500
    eoi_frame = ignition_frame + 0.5*fps # end of ignition period of interest
    eow_frame = ignition_frame + 0.05*fps # end of weighted frames
    flame_step = 0
    print('threshold value: ',thresh)
    for j in frames:
        j = j.astype(float)
        arr = (j-thresh)>0
        if map_type == 'ig' and frame_num >= ignition_frame:
            heatmap = map_cumulative(heatmap,arr,frame_num,eow_frame,scalar=50) # when scalar is greater than 1, this will weight the frames before eow_frames by the scaled amount
        elif map_type == 'dis_ig' and frame_num >= ignition_frame:
            heatmap,flame_step = map_discrete(heatmap,arr,flame_step,spacer=5) # the spacer helps set the discrete layers apart more or less (how much the color varies for each time step)
        elif map_type == 'dis_c' and frame_num >= ignition_frame:
            heatmap,flame_step = map_discrete(heatmap,arr,flame_step,spacer=5)
        elif map_type == 'flaming':
            if frame_num >= ignition_frame:
                heatmap = map_cumulative(heatmap,arr,frame_num,eow_frame,scalar=1)
            else:
                frame_num += 1
                continue
        else:
            heatmap = map_cumulative(heatmap,arr,frame_num,eow_frame,scalar=1)

        if map_type == 'preig' and frame_num >= ignition_frame:
            break
        elif map_type == 'all' and frame_num > test.eof:
            break
        elif map_type == 'ig' and frame_num >= eoi_frame:
            break
        elif map_type == 'dis_ig' and frame_num >= eoi_frame:
            break
        elif map_type == 'dis_c' and frame_num >= test.eof:
            break
        elif map_type == 'flaming' and frame_num >= test.eof:
            break
        frame_num += 1
    if save is True:
        np.save(savepath,heatmap)
    return None

def get_mapsets_d(test,thresh,save):
    print(test.testnumber)
    name = test.filename.replace('.tif','')
    cwd = os.getcwd()
    filepath_check = cwd+'_cache\\heatmaps\\dsets\\' + name + '_dsets01.npy'
    # filepath_check = cwd+'_cache\\heatmaps\\dsets2\\' + name + '_dsets01.npy'
    ischeck = checkfile(filepath_check,test,checktype=True,isinput=True)
    if ischeck == False:
        return
    filepath = os.getcwd().replace('wfire','') + test.filename
    img = readfile(filepath,True)[0]
    frames,num_frames,num_rows,num_cols = get_image_properties(img)
    ignition_frame = test.ignition_time[1]
    fps = 500
    item = 1
#---------------------
    time_step = 0.1*fps
    # time_step = 0.42*fps
    flame_step = 0
    for i in range(7):
        heatmap = np.zeros((num_rows,num_cols))
        if i == 6:
            cut_start = int(ignition_frame)
            flame_step = 0
        else:
            cut_start = int(ignition_frame+time_step*i)
            cut_stop = int(cut_start+time_step)
        frames_oi = frames[cut_start:cut_stop]
        for j in frames_oi:
            j = j.astype(float)
            arr = (j-thresh)>0
            heatmap,flame_step = map_discrete(heatmap,arr,flame_step,spacer=5)
        if save is True:
            savepath = cwd+'_cache\\heatmaps\\dsets\\' + name + '_dsets0' + str(item) + '.npy'
            # savepath = cwd+'_cache\\heatmaps\\dsets2\\' + name + '_dsets0' + str(item) + '.npy'
            np.save(savepath,heatmap)
        item+=1
    return None

def get_mapsets_c(test,thresh,save,maptag):
    print(test.testnumber)
    name = test.filename.replace('.tif','')
    cwd = os.getcwd()
    if maptag == 'alpha':
        filepath_check = cwd+'_cache\\heatmaps\\csets\\' + name + '_csets01.npy'
    else:
        filepath_check = cwd+'_cache\\heatmaps\\cbsets\\' + name + '_cbsets01.npy'
    ischeck = checkfile(filepath_check,test,checktype=True,isinput=True)
    if ischeck == False:
        return
    filepath = os.getcwd().replace('wfire','') + test.filename
    img = readfile(filepath,True)[0]
    frames,num_frames,num_rows,num_cols = get_image_properties(img)
    ignition_frame = test.ignition_time[1]

    # print(filepath)
    # print(ignition_frame)
    # input('checking....')

    fps = 500
    item = 1
    cut_start = int(ignition_frame)
    if maptag == 'alpha':
        stop_times_s = [0.5,1,2] # seconds after ignition
        stop_times = np.multiply(stop_times_s,fps)+ignition_frame
        eow_frame = cut_start+(0.05*fps)
        scalar = stop_times_s[-1]*fps
        for i in range(len(stop_times)):
            heatmap = np.zeros((num_rows,num_cols))
            cut_stop = int(stop_times[i])
            frames_oi = frames[cut_start:cut_stop]
            frame_num = cut_start
            for j in frames_oi:
                j = j.astype(float)
                arr = (j-thresh)>0
                heatmap = map_cumulative(heatmap,arr,frame_num,eow_frame,scalar)
                frame_num+=1
            if save is True:
                savepath = cwd+'_cache\\heatmaps\\csets\\' + name + '_csets0' + str(item) + '.npy'
                np.save(savepath,heatmap)
            item+=1
    elif maptag == 'beta':
        stop_time = int((test.eof-ignition_frame)/2)+ignition_frame
        stop_time = int(test.eof)
        eow_frames_s = [0.01,0.05,0.1,0.2]
        eow_frames = [cut_start+(eow_frames_s[0]*fps),cut_start+(eow_frames_s[1]*fps),cut_start+(eow_frames_s[2]*fps),cut_start+(eow_frames_s[3]*fps)]
        relax = 1.5
        scalar = (stop_time-ignition_frame)*(1+relax)
        scalars = [scalar,scalar*0.2,scalar*0.1,scalar*0.05]
        frames_oi = frames[cut_start:stop_time]
        for i in range(len(eow_frames)):
            heatmap = np.zeros((num_rows,num_cols))
            frame_num = cut_start
            for j in frames_oi:
                j = j.astype(float)
                arr = (j-thresh)>0
                heatmap = map_cumulative(heatmap,arr,frame_num,eow_frame=eow_frames[i],scalar=scalars[i])
                frame_num+=1
            if save is True:
                savepath = cwd+'_cache\\heatmaps\\cbsets\\' + name + '_cbsets0' + str(item) + '.npy'
                np.save(savepath,heatmap)
            item+=1
    return

def map_discrete(heatmap,arr,flame_step,spacer):
    heatmap_new = heatmap > 0
    arr_new = (arr*1)-heatmap_new
    arr_new_bool = arr_new > 0
    arr_add = arr_new_bool*(flame_step*spacer)
    heatmap+=arr_add
    flame_step+=1
    return heatmap,flame_step

def map_cumulative(heatmap,arr,frame_num,eow_frame,scalar):
    if frame_num <= eow_frame and scalar != 1:
        arr = arr*scalar
    heatmap+=arr
    return heatmap

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

def displaymaps(heatmap,map_type,cmap_usr,**kwargs):
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
    if 'xvals' in kwargs:
        plt.plot(kwargs['xvals'],kwargs['yvals'])
    plt.imshow(imgs,cmap=cmap_usr)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    # if map_type == 'ig':
    #     plt.colorbar()
    show_window(noticks=True,winmax=True,closewin=True,showwin=True)
    return True

def get_points(img,test,points_type):
    """....
    """
    cwd = os.getcwd()
    if points_type == 'profile':
        num_points = 2
        pfilepath = cwd+'\\points\\'+test.filename.replace('.tif','')+'_points.txt'
    elif points_type == 'timeline':
        num_points = 20
        pfilepath = cwd+'\\points_timeline\\'+test.filename.replace('.tif','')+'_points_timeline.txt'
    elif points_type == 'vert':
        num_points = 1
        pfilepath = cwd+'\\points_vert\\'+test.filename.replace('.tif','')+'_points_vert.txt'
    elif points_type == 'grid':
        pfilepath = cwd+'\\points_grid\\'+test.filename.replace('.tif','')+'_points_grid.txt'
        num_points = 2
    elif points_type == 'selectarea':
        pfilepath = cwd + '_cache\\flame_area\\selectarea\\'+test.filename.replace('.tif','_selectpoints.npy')
        num_points = None
    elif points_type == 'flameheight':
        pfilepath = cwd + '_cache\\flame_height\\'+test.filename.replace('.tif','_flameheightpoints.npy')
        num_points = None
    elif points_type == 'intermittancy':
        pfilepath = cwd + '_cache\\points\\intermittancy\\'+test.filename.replace('.tif','_points_intermittancy.txt')
        num_points = None
    elif points_type == 'crop':
        pfilepath = cwd+'_cache\\points\\crop\\'+test.filename.replace('.tif','')+'_points_crop.txt'
        num_points = None
    ischeck = checkfile(pfilepath,test,checktype=True,isinput=True)
    if ischeck == False:
        return 0,0,False
    num_rows = img.shape[0]
    calib = np.zeros((num_rows,1))
    calib[0,0] = 4500
    if points_type != 'selectarea':
        img_new = np.concatenate((img,calib),axis=1)
    if points_type == 'selectarea' or points_type == 'flameheight' or points_type == 'intermittancy' or points_type == 'crop':
        # plt.imshow(img,cmap='nipy_spectral_r')
        plt.imshow(img)
    else:
        plt.imshow(img_new)
    plt.get_current_fig_manager().window.showMaximized()
    if points_type == 'grid' or points_type == 'selectarea':
        p = plt.ginput(n=-1,timeout=-1,show_clicks=True)
        rows,cols = img.shape[0],img.shape[1]
        p = refine_gridpoints(p,rows,cols)
    else:
        p = plt.ginput(num_points,timeout=-1)
        num_points = len(p)
        print(num_points)
    plt.close()

    return p,num_points

def refine_gridpoints(p,rows,cols):
    x_left,x_right,y_bot,y_top = cols-1,0,0,rows-1
    for k in range(0,len(p)):
        if p[k][0] < x_left:
            x_left = int(p[k][0])
        if p[k][0] > x_right:
            x_right = int(p[k][0])
        if p[k][1] > y_bot:
            y_bot = round(p[k][1])
        if p[k][1] < y_top:
            y_top = int(p[k][1])
    p_new = [(x_left,x_right),(y_bot,y_top)]
    return p_new

def change_tests():
    """....
    """
    sets = []
    usr = input('Choose from preset selections (y/n)')
    if usr == 'y':
        sets = choose_preset(sets)
        return sets
    n = input('How many sets do you want to specify? - ')
    for i in range(0,int(n)):
        m1 = input('Enter start number of set: ')
        m2 = input('Enter end number of set: ')
        sets.append(int(m1))
        sets.append(int(m2))
    return sets

def choose_preset(sets):
    running = True
    is_1,is_2,is_3,is_4,is_5,is_6,is_7,is_8=0,0,0,0,0,0,0,0
    while running:
        os.system('cls')
        print('\n\n\n\n\n','--------------------------------------------------','\n')
        print('Individual (65mm): 1')
        print('Individual (83mm): 2')
        print('Multiple-Live (65mm): 3')
        print('Multiple-Live (83mm): 4')
        print('Multiple-dried (65mm): 5')
        print('Multiple-dried (83mm): 6')
        print('Multiple-subset py: 7')
        print('Single Douglas-fir needles: 8')
        print('Selection finished: f')
        usr = input('Choose selection: ')
        if usr == '1':
            if is_1:
                input('This has already been selected, press enter to continue')
                continue
            sets.append([189,198,119,128,249,258,89,98,269,278])
            is_1 = True
        elif usr == '2':
            if is_2:
                input('This has already been selected, press enter to continue')
                continue
            sets.append([199,202,129,138,259,268,99,108,279,288])
            is_2 = True
        elif usr == '3':
            if is_3:
                input('This has already been selected, press enter to continue')
                continue
            sets.append([209,218,169,178,229,238,159,168,289,298])
            is_3 = True
        elif usr == '4':
            if is_4:
                input('This has already been selected, press enter to continue')
                continue
            sets.append([219,228,179,188,239,248,149,158,299,308])
            is_4 = True
        elif usr == '5':
            if is_5:
                input('This has already been selected, press enter to continue')
                continue
            sets.append([311,320,323,332,335,344,347,356,359,368])
            is_5 = True
        elif usr == '6':
            if is_6:
                input('This has already been selected, press enter to continue')
                continue
            sets.append([369,378,379,388,389,398,399,408,409,418])
            is_6 = True
        elif usr == '7':
            if is_7:
                input('This has already been selected, press enter to continue')
                continue
            sets.append([420,429,430,439,440,449])
            is_7 = True
        elif usr == '8':
            if is_8:
                input('This has already been selected, press enter to continue')
                continue
            sets.append([450,454,455,459,460,464])
            is_8 = True
        elif usr == 'f':
            running = False
    sets_new = []
    for i in range(len(sets)):
        var = sets[i]
        for j in range(len(var)):
            sets_new.append(var[j])
    return sets_new

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

def load_heatmap(test,map_type):
    """...
    """
    pathname = test.filename.replace('.tif','')
    cwd = os.getcwd()
    if map_type == 'all':
        path = cwd+'\\heatmaps\\'+pathname+'_heatmap.npy'
    elif map_type == 'preig':
        path = cwd+'_cache\\heatmaps\\preig\\'+pathname+'_preig_heatmap.npy'
    elif map_type == 'ig':
        path = cwd+'_cache\\heatmaps\\ig\\'+pathname+'_ig_heatmap.npy'
    elif map_type == 'dis_ig':
        path = cwd+'_cache\\heatmaps\\dis_ig\\'+pathname+'_disig_heatmap.npy'
    elif map_type == 'dis_c':
        path = cwd+'_cache\\heatmaps\\dis_c\\'+pathname+'_disc_heatmap.npy'
    elif map_type == 'flaming':
        path = cwd+'_cache\\heatmaps\\flaming\\'+pathname+'_flaming_heatmap.npy'
    else:
        return None
    msg = 'You can create this file by generating a heatmap'
    ischeck = checkfile(path,test,checktype=False,isinput=True)
    if ischeck == False:
        return None
    heatmap = np.load(path)
    return heatmap

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



def save_points(test,p,num_points,points_type,**kwargs):
    """...
    """
    cwd = os.getcwd()
    if points_type == 'profile':
        pfilepath = cwd+'\\points\\'+test.filename.replace('.tif','')+'_points.txt'
    elif points_type == 'timeline':
        pfilepath = cwd+'\\points_timeline\\'+test.filename.replace('.tif','')+'_points_timeline.txt'
    elif points_type == 'vert':
        pfilepath = cwd+'\\points_vert\\'+test.filename.replace('.tif','')+'_points_vert.txt'
    elif points_type == 'grid':
        pfilepath = cwd+'\\points_grid\\'+test.filename.replace('.tif','')+'_points_grid.txt'
    elif points_type == 'flame_height':
        pfilepath = cwd+'_cache\\points\\flame_height\\'+test.filename.replace('.tif','')+'_points_height.txt'
        # input('Error (press \'Enter\' to return)')
        # return
    elif points_type == 'intermittancy':
        pfilepath = cwd+'_cache\\points\\intermittancy\\'+test.filename.replace('.tif','')+'_points_intermittancy.txt'
        if 'pathname' in kwargs:
            pfilepath = kwargs['pathname']
    elif points_type == 'crop':
        pfilepath = cwd+'_cache\\points\\crop\\'+test.filename.replace('.tif','')+'_points_crop.txt'
    ischeck = checkfile(pfilepath,test,checktype=True,isinput=True)
    if ischeck == False:
        return
    pfile = open(pfilepath,'x')
    if points_type == 'profile':
        pfile.write(str(p[0][0])+' '+str(p[0][1])+' '+str(p[1][0])+' '+str(p[1][1]))
    else:
        for i in range(num_points):
            for j in range(2):
                pfile.write(str(p[i][j])+' ')
    pfile.close
    print('Save process complete (Press enter)')

def get_line_coordinates(test,d):
    coordinates = np.zeros((1,4))
    conv_fact = test.spatial_calibration
    for ii in range(0,len(d)):
        d[ii] /= 100
        d[ii] /= conv_fact
        d[ii] = round(d[ii])
    pathname = test.filename.replace('.tif','')
    cwd = os.getcwd()
    path = cwd+'\\points\\'+pathname+'_points.txt'
    ischeck = checkfile(path,test,checktype=False,isinput=True)
    if ischeck == False:
        return 0
    p = np.loadtxt(path,unpack=True)
    L = d[1]+d[2]
    x_0 = (p[0]+p[2])/2
    y_0 = (p[1]+p[3])/2
    center = [x_0,y_0]

    x_c = center[0]
    y_c = center[1]-d[0]
    x_L = x_c - d[1]
    x_r = x_c + d[2]
    dx = (x_r-x_L)/100
    coordinates[0,0],coordinates[0,1],coordinates[0,2],coordinates[0,3] = y_c,x_L,1,L
    for ii in range(0,len(d)):
        d[ii] *= 100
        d[ii] *= conv_fact
        d[ii] = round(d[ii])
    return coordinates

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

def change_linepar(distance):
    print('Distance is currently [vert distance, distance to left, distance to right] (cm) - ',distance,'\n')
    usr = input('Would you like the line to be above or below? (a/b)')
    if usr == 'a':
        dirc = 1
    elif usr == 'b':
        dirc = -1
    else:
        return distance
    distance[0] = int(input('What vertical distance would you like to set? (cm): '))
    distance[0] = distance[0] * dirc
    return distance

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
            if coordinates == 0:
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
        if isnorm == True:
            plt.plot(x_plot,line_avg_norm.T,label=labels[int(i/2)])
            axlabel = 's/s'
        elif isnorm == False:
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
        
def change_ylim(ylim):
    ylim = input('Set ylim for plotting: ')
    return ylim

def create_line(x_vals,y_vals,ind):
    # print(x_vals,y_vals)
    p1 = [x_vals[ind],y_vals[ind]]
    p2 = [x_vals[ind+1],y_vals[ind+1]]
    diff = p2[0]-p1[0]
    if diff < 0:
        line_type = 'backwards'
    elif diff > 0:
        line_type = 'forwards'
    elif diff == 0:
        p2[0]*=1.01
        diff = p2[0]-p1[0]
        line_type = 'forwards'
    if abs(int(diff)) < 2:
        diff = 2
    xpoints,ypoints = np.array(()),np.array(())
    m = (p2[1]-p1[1])/(p2[0]-p1[0])
    b = p2[1]-m*p2[0]
    for i in range(abs(int(diff))):
        if line_type == 'forwards':
            x = int(p1[0]+i)
        elif line_type == 'backwards':
            x = int(p1[0]-1)
        y = int(m*x+b)
        xpoints = np.append(xpoints,x)
        ypoints = np.append(ypoints,y)
    # print(p1,p2)
    # print(diff)
    # print(abs(int(diff)))
    # print(xpoints,ypoints)
    # input()
    line = Line(p1,p2,line_type,xpoints,ypoints,m,b)
    return line

def get_intermittancy(test):
    threshold = get_threshold(test,fmc=test.fmc,tag='other')
    cwd = os.getcwd()
    pfilepath = cwd+'_cache\\points\\intermittancy\\'+test.filename.replace('.tif','')+'_points_intermittancy.txt'
    ischeck = checkfile(pfilepath,test,checktype=False,isinput=False)
    ignition_frame = int(test.ignition_time[1])
    eof = int(test.eof)
    frame_span = eof-ignition_frame
    file = os.getcwd().replace('wfire','') + test.filename
    img,filename = readfile(file,True)
    frames,num_frames,num_rows,num_cols = get_image_properties(img)
    quart = round(frame_span/4)
    intermittancy = np.zeros((quart,2,4))
    x_time = np.linspace(0,1,quart)

    # heatmap = load_heatmap(test,map_type='all')
    for ii in range(4):
        quart_num = ii+1
        heatmap = get_quartermap(frames[ignition_frame+quart*ii:ignition_frame+quart*(ii+1)],num_rows,num_cols,threshold)

        pfilepath = cwd+'_cache\\points\\intermittancy\\'+test.filename.replace('.tif','')+'_points_intermittancy_0'+str(quart_num)+'.txt'
        ischeck = checkfile(pfilepath,test,checktype=False,isinput=False)
        if ischeck == False:
            p,num_points = get_points(heatmap,test,points_type='intermittancy')
            save_points(test,p,num_points,'intermittancy',pathname=pfilepath)
            p = np.loadtxt(pfilepath,unpack=True)
        else:
            p = np.loadtxt(pfilepath,unpack=True)
            num_points = int(len(p)/2)
        x_val,y_val = [],[]
        x_above,y_above = [],[]
        x_below,y_below = [],[]
        for i in range(0,len(p),2):
            x_val.append(p[i])
            y_val.append(p[i+1])
        mid_point = int(x_val[0]),int(y_val[0])
        for i in range(len(y_val)):
            if i == 0:
                continue
            if y_val[i] <= mid_point[1]:
                x_above.append(x_val[i])
                y_above.append(y_val[i])
            else:
                x_below.append(x_val[i])
                y_below.append(y_val[i])
        top_row,bottom_row = min(y_val),max(y_val)

        num_lines = [int(len(x_above)-1),int(len(x_below)-1)]
        lines_above,lines_below = [],[]
        for i in range(num_lines[0]):
            line = create_line(x_above,y_above,i)
            lines_above.append(line)
        for i in range(num_lines[1]):
            line = create_line(x_below,y_below,i)
            lines_below.append(line)
        # print(min(x_above),max(x_above))
        # print(min(x_below),max(x_below))
        # input()

        for i in range(quart):
            ref_frame = frames[ignition_frame+quart*ii+i].astype(float)
            for j in range(0,mid_point[1]):
                if j < top_row:
                    # heatmap[j,:] = heatmap.max()
                    if ref_frame[j].max() >= threshold:
                        intermittancy[i][0][ii] = True
                        break
                    else:
                        continue
                elif j >= top_row and j < mid_point[1]:
                    for k in range(num_cols):
                        pix = ref_frame[j,k]
                        if pix < threshold:
                            continue
                        if k < min(x_above) or k > max(x_above):
                            # heatmap[j,k] = heatmap.max()
                            # continue
                            intermittancy[i][0][ii] = True
                            break
                        m,b,line_type = check_line(k,ref_frame,x_above,y_above,lines_above)
                        xcheck,ycheck = k,j
                        yactual = m*xcheck+b
                        heatmap[int(yactual),k] = heatmap.max()
                        if line_type == 'forward':
                            if ycheck > yactual:
                                continue
                            else:
                                # heatmap[int(ycheck),k] = heatmap.max()
                                intermittancy[i][0][ii] = True
                                break
                        elif line_type == 'backwards':
                            if ycheck < yactual:
                                continue
                            else:
                                intermittancy[i][0][ii] = True
                                break
                    break
            for j in range(mid_point[1],num_rows):  
                if j >= mid_point[1] and j < bottom_row:
                    for k in range(num_cols):
                        pix = ref_frame[j,k]
                        if pix < threshold:
                            continue
                        if k < min(x_below) or k > max(x_below):
                            # heatmap[j,k] = heatmap.max()
                            # continue
                            intermittancy[i][1][ii] = True
                            break
                        m,b,line_type = check_line(k,ref_frame,x_below,y_below,lines_below)
                        xcheck,ycheck = k,j
                        yactual = m*xcheck+b
                        # heatmap[int(yactual),k] = heatmap.max()
                        if ycheck < yactual:
                            continue
                        else:
                            # heatmap[int(ycheck),k] = heatmap.max()
                            intermittancy[i][1][ii] = True
                            break
                    break
                elif j >= bottom_row:
                    # heatmap[j,:] = heatmap.max()
                    if ref_frame[j].max() >= threshold:
                        intermittancy[i][1][ii] = True
                        heatmap[j,:] = heatmap.max()
                        break
                    else:
                        continue
        q = [intermittancy[:,0,ii],intermittancy[:,1,ii]]
        q_sum = [round(np.sum(q[0])/quart,3),round(np.sum(q[1])/quart,3)]  
        print('Quarter ',quart_num) 
        print(q_sum)
    print(frame_span,quart)
    input('press enter')
    for i in range(4):
        plt.figure()
        plt.plot(x_time,intermittancy[:,0,i])
        show_window(noticks=False,winmax=False,closewin=False,showwin=True)
        plt.figure()
        plt.plot(x_time,intermittancy[:,1,i])
        show_window(noticks=False,winmax=False,closewin=True,showwin=True)

def check_line(col,ref_frame,xvals,yvals,lines):
    loi = []
    for i in lines:
        xpoints,ypoints = i.xpoints,i.ypoints
        # print(xpoints,col)
        num = np.where(xpoints==col)[0].shape[0]
        if num != 0:
            loi.append([i,xpoints,ypoints])
    if len(loi) == 0:
        for i in lines:
            xpoints,ypoints = i.xpoints,i.ypoints
            num01 = np.where(xpoints==col+1)[0].shape[0]
            num02 = np.where(xpoints==col-1)[0].shape[0]
            # print(num01,num02)
            if num01 != 0 or num02 != 0:
                loi.append([i,xpoints,ypoints])
    dis = 9999
    m = loi[0][0].m
    b = loi[0][0].b
    line_type = loi[0][0].line_type
    for j in range(len(loi)):
        xpoints,ypoints = loi[j][1],loi[j][2]
        for k in range(len(xpoints)-1):
            dis_new = np.sqrt((xpoints[k+1]-xpoints[k])**2+(ypoints[k+1]-ypoints[k])**2)
            if dis_new < dis:
                m = loi[j][0].m
                b = loi[j][0].b
                line_type = loi[j][0].line_type
                dis = dis_new
    return m,b,line_type

    # change = False
    # for ii in range(len(xvals)):
    #     if ii != len(xvals)-1:
    #         if xvals[ii] == xvals[ii+1]:
    #             xvals[ii+1]*=1.01
    #             change = True
    #     # print(len(xvals),k,ii,ii+1)
    #     # ref_frame[yvals[ii],xvals[ii]] = ref_frame.max()
    #     # ref_frame[:,k] = ref_frame.max()
    #     # plt.imshow(ref_frame)
    #     # plt.show()
    #     if k >= xvals[ii] and k <= xvals[ii+1]:
    #         x1,y1 = xvals[ii],yvals[ii]
    #         x2,y2 = xvals[ii+1],yvals[ii+1]
    #         if change:
    #             change = False
    #             xvals[ii+1]/=1.01
    #         break
    #     else:
    #         if change:
    #             change = False
    #             xvals[ii+1]/=1.01
    # m = (y2-y1)/(x2-x1)
    # b = y2-m*x2
    # return m,b
    #     # xcheck,ycheck = k,j
    #     # yactual = m*xcheck+b
    #     # if ycheck > yactual:
    #     #     continue
    #     # else:


def get_flametimeline(test):        
    cwd = os.getcwd()
    pfilepath = cwd+'\\points_timeline\\'+test.filename.replace('.tif','')+'_points_timeline.txt'
    ischeck = checkfile(pfilepath,test,checktype=False,isinput=True)
    if ischeck == False:
        return
    p = np.loadtxt(pfilepath,unpack=True)
    num_points = int(len(p)/2)
    x_val = []
    y_val = []
    for jj in range(0,len(p),2):
        x_val.append(int(p[jj]))
        y_val.append(int(p[jj+1]))
    file = os.getcwd().replace('wfire','') + test.filename
    img,filename = readfile(file,True)
    frames,num_frames,num_rows,num_cols = get_image_properties(img)
    ignition_frame = int(test.ignition_time[1])
    eof = int(test.eof)
    frame_span = int(eof-ignition_frame)
    timeline = np.zeros((num_points,frame_span))
    x_time = np.linspace(0,frame_span,frame_span)/frame_span
    save_filepath = cwd+'\\plots_timeline\\'+test.filename.replace('.tif','')+'_timeline.png'
    tfilepath = cwd+'\\data_timeline\\'+test.filename.replace('.tif','')+'_timeline.txt'
    tfile = open(tfilepath,'x')
    for jjj in range(num_points):
        if jjj <= num_points/2:
            color = 'k'
            level = 1
        else:
            color = 'b'
            level = -1
        for k in range(frame_span):
            frame = frames[k+ignition_frame].astype(float)
            if frame[y_val[jjj],x_val[jjj]] > 35:
                timeline[jjj,k] = level
                save_val = level*x_time[k]
                tfile.write(str(save_val)+'\n')
        plt.plot(x_time,timeline[jjj,:].T,color)
    tfile.close()    
    # plt.show()
    # if os.path.exists(save_filepath):
    #     print('---------------------------------------------------------\nLooks like there\'s already a file with this name.\nDelete existing file if you are wanting to overwrite\n')
    #     print(pfilepath)
    #     usr = input('Ok (press return)')
    #     return
    # plt.savefig(save_filepath)
    plt.close()
            
def calc_avgint(test,args):
    """ Calculates the average light intensity (for pixels above threshold)
    """
    threshold = args[0]
    check = True
    file = os.getcwd().replace('wfire','') + test.filename
    img,filename = readfile(file,True)
    frames,num_frames,num_rows,num_cols = get_image_properties(img)
    avgint = np.zeros((num_frames,1))
    x_left,x_right,y_bot,y_top = load_gridpoints(test)


    counter = 0
    cat = [0,0,0]
    for k in range(0,num_frames):
        counter+=1
        ref_frame = frames[k]
        ref_frame = ref_frame.astype(float) # Change to float to handle subtraction correctly
        ref_frame[y_top:y_bot,x_left:x_right] = 0
        if ref_frame.max() < threshold:
            continue
        x_bool = ((ref_frame-threshold)>=0)
        numpixels_frame = x_bool.sum()
        y_mod = np.multiply(ref_frame,x_bool)
        total = y_mod.sum()
        avgint[k,0] = total/numpixels_frame/255
        if avgint[k,0] >= 0 and avgint[k,0] < 85:
            cat[0]+=1
        elif avgint[k,0] >= 85 and avgint[k,0] < 170:
            cat[1]+=1
        elif avgint[k,0] >= 170 and avgint[k,0] <= 255:
            cat[2]+=1
        else:
            input('Error')
        if check == False:
            plt.imshow(y_mod,cmap='nipy_spectral_r')
            plt.get_current_fig_manager().window.state('zoomed')
            plt.show(block=False)
            plt.pause(0.5)
            plt.clf
            if counter%10 == 0:
                plt.close()
            try:
                if keyboard.is_pressed('q'):
                    check = True
            except:
                print()
    x_plot = np.linspace(1,num_frames,num_frames)
    x_plot/=num_frames
    plt.plot(x_plot,avgint)
    plt.xlabel('Normalized burning time')
    plt.ylabel('Normalized average flame intensity')
    plt.ylim(0,1)
    save_filepath = os.getcwd()+'\\plots_avgint\\'+test.filename.replace('.tif','')+'_avgint.png'
    ischeck = checkfile(save_filepath,test,checktype=True,isinput=True)
    if ischeck == False:
        return
    plt.savefig(save_filepath)
    plt.close()
    return

def creategrids(test):
    sector_width = 30
    heatmap = load_heatmap(test,'all')
    cwd = os.getcwd()
    pfilepath = cwd+'\\points_grid\\'+test.filename.replace('.tif','')+'_points_grid.txt'
    ischeck = checkfile(pfilepath,test,checktype=False,isinput=True)
    if ischeck == False:
        return 999
    p = np.loadtxt(pfilepath)
    x_left,x_right,y_bot,y_top = p[0],p[1],p[2],p[3]
    img = heatmap
    num_rows = img.shape[0]
    calib = np.zeros((num_rows,1))
    calib[0,0] = 4500
    img = np.concatenate((img,calib),axis=1)

    num_x = int((x_right-x_left)/sector_width)+2
    num_y = int((y_bot-y_top)/sector_width)+2
    
    x_ticks,y_ticks = [],[]
    for k in range(0,num_x):
        x = k*sector_width+x_left
        x_ticks.append(x)
        if x >= x_right:
            x_ticks[k] = x_right
            break
    for k in range(0,num_y):
        y = k*sector_width+y_top 
        y_ticks.append(y)
        if y >= y_bot:
            y_ticks[k] = y_bot
            break
    x_plot = np.linspace(x_left,x_right,100)
    plt.imshow(img,cmap='nipy_spectral_r')
    plt.get_current_fig_manager().window.showMaximized()
    for k in y_ticks:
        y_plot = np.linspace(k,k,100)
        plt.plot(x_plot,y_plot,'k')
    y_plot = np.linspace(y_top,y_bot,100)
    for k in x_ticks:
        x_plot = np.linspace(k,k,100)
        plt.plot(x_plot,y_plot,'k')
    show_window(noticks=True,winmax=False,closewin=True,showwin=True)
            
    return
                
                
def get_median(test):
    cwd = os.getcwd()
    filepath = cwd+'\\data_timeline\\'+test.filename.replace('.tif','')+'_timeline.txt'
    ischeck = checkfile(filepath,test,checktype=False,isinput=True)
    if ischeck == False:
        return None
    data = np.loadtxt(filepath,unpack=True)
    data = np.abs(data)
    med_data = np.array(())
    vals = 0
    for i in range(0,len(data)):
        if vals == 0:
            med_data = np.append(med_data,data[i])
            vals+=1
        elif vals != 0:
            arr = (med_data-data[i])==0
            isthere = np.any(arr)
            if isthere==False:
                med_data = np.append(med_data,data[i])
                vals+=1
        else:
            continue
    median = np.median(med_data)
    return median

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

def get_max_flame_area(test):
    isinput=False
    afilepath = os.getcwd()+'_cache\\flame_area\\vals\\'+test.filename.replace('.tif','_area_vals.npy')
    mfilepath = os.getcwd()+'_cache\\flame_area\\frames\\'+test.filename.replace('.tif','_areaframe_cropped.npy')
    ischeck_m,ischeck_a = checkfile(mfilepath,test,True,isinput),checkfile(afilepath,test,True,isinput)
    if ischeck_m == False or ischeck_a == False:
        return None
    x_left,x_right,y_bot,y_top = load_gridpoints(test)
    if x_left == None:
        input('No gridpoints')
        return 999
    file = os.getcwd().replace('wfire','') + test.filename
    img,filename = readfile(file,True)
    frames,num_frames,num_rows,num_cols = get_image_properties(img)
    fmc = test.fmc
    threshold = get_threshold(test,test.fmc,'other')

    max_flame_area,m_frame = 0,0
    pixel_length = test.spatial_calibration
    npf = 0
    for k in range(num_frames-1):
        ref_frame = frames[k]
        ref_frame = ref_frame.astype(float) # Change to float to handle subtraction correctly
        ref_frame[y_top:y_bot,x_left:x_right] = 0
        if ref_frame.max() < threshold:
            continue
        flame_area = calc_area(ref_frame,None,threshold,pixel_length,'frame')
        if flame_area > max_flame_area:
            max_flame_area = flame_area
            m_frame = ref_frame
            frame_num = k

    print('Threshold: ',threshold)
    print('Total area: ',round(max_flame_area,2),' cm^2')
    np.save(mfilepath,m_frame)
    np.save(afilepath,[max_flame_area,frame_num])

def load_area(test):
    newvals = True
    afilepath = os.getcwd()+'_cache\\flame_area\\vals\\'+test.filename.replace('.tif','_area_vals.npy')
    ischeck = checkfile(afilepath,test,checktype=False,isinput=True)
    if ischeck == False:
        return None,None
    vals = np.load(afilepath)
    max_flamearea,frame_num = vals[0]/(10**2),vals[1]

    areaframe_ell_path = os.getcwd()+'_cache\\flame_area\\frames\\'+test.filename.replace('.tif','_areaframe_ell.npy')
    if os.path.exists(areaframe_ell_path) and newvals:
        threshold = get_threshold(test,test.fmc,'area')
        areas = calc_area(None,areaframe_ell_path,threshold,test.spatial_calibration,'file')
        if isinstance(areas,list):
            if areas[0] > areas[1]:
                max_flamearea = areas[0]
            else:
                max_flamearea = areas[1]
        else:
            max_flamearea = areas
    else:
        print('\nTest number ',test.testnumber,' is using an outdated area')
    return max_flamearea,frame_num     

def calc_area(ref_frame,filename,threshold,pixel_length,tag):
    check_frames = False       
    pixel_area = pixel_length**2
    if tag == 'file':
        ref_frame = np.load(filename)
    ref_frame = ref_frame.astype(float)
    rows,cols = ref_frame.shape[0],ref_frame.shape[1]
    if rows != cols:
        ref_frame01,ref_frame02 = ref_frame[0:rows,0:int(cols/2)],ref_frame[0:rows,int(cols/2):cols]
        x_bool01,x_bool02 = ((ref_frame01-threshold)>=0),((ref_frame02-threshold)>=0)
        if check_frames:
            dis_frame01 = np.multiply(ref_frame01,x_bool01)
            dis_frame02 = np.multiply(ref_frame02,x_bool02)
            dis_show = np.concatenate((dis_frame01,dis_frame02),axis=1)
            plt.imshow(dis_show,cmap='nipy_spectral_r')
            plt.title('Checking frame')
            show_window(noticks=True,winmax=True,closewin=True)
        numpixels_frame01,numpixels_frame02 = x_bool01.sum(),x_bool02.sum()
        if numpixels_frame01 > numpixels_frame02:
            numpixels_frame = numpixels_frame01
        else:
            numpixels_frame = numpixels_frame02
        flame_area01,flame_area02 = pixel_area*numpixels_frame01*(100**2),pixel_area*numpixels_frame02*(100**2)
        return [flame_area01,flame_area02]
    else:
        x_bool = ((ref_frame-threshold)>=0)
        if check_frames:
            dis_show = np.multiply(ref_frame,x_bool)
            plt.imshow(dis_show,cmap='nipy_spectral_r')
            plt.title('Checking frame')
            show_window(noticks=True,winmax=True,closewin=True)
        numpixels_frame = x_bool.sum()
        flame_area = pixel_area*numpixels_frame*(100**2)
        return flame_area

def get_threshold(test,fmc,tag):
    fpath = os.getcwd()+'_cache\\adjusted_thresholds.npy'
    if tag == 'area'and os.path.exists(fpath):
        fpath = os.getcwd()+'_cache\\adjusted_thresholds.npy'
        thresholds = np.load(fpath)
        threshold = thresholds[int(test.testnumber-1)]
    else:
        if fmc == 0:
            threshold = 50
        elif fmc > 0:
            threshold = 35
        elif fmc < 0:
            threshold = 100
    return threshold

def get_ima(test):
    check_frames = False
    mfilepath = os.getcwd()+'_cache\\flame_area\\frames\\'+test.filename.replace('.tif','_areaframe_cropped.npy')
    areaframe_ell_path = os.getcwd()+'_cache\\flame_area\\frames\\'+test.filename.replace('.tif','_areaframe_ell.npy')
    if os.path.exists(areaframe_ell_path):
        areaframe_ell = np.load(areaframe_ell_path)
        m_frame = areaframe_ell
        m_frame = m_frame.astype(float)
        threshold = get_threshold(test,None,'area')
        # input(threshold)
    else:
        print('Using outdated data')
        ischeck = checkfile(mfilepath,test,checktype=False,isinput=True)
        if ischeck == False:
            return 999
        m_frame = np.load(mfilepath)
        fmc = test.fmc
        threshold = get_threshold(test,fmc,'other')

    rows,cols = m_frame.shape[0],m_frame.shape[1]
    if rows != cols:
        m_frame01,m_frame02 = m_frame[0:rows,0:int(cols/2)],m_frame[0:rows,int(cols/2):cols]
        x_bool01,x_bool02 = ((m_frame01-threshold)>=0),((m_frame02-threshold)>=0)
        if check_frames:
            dis_frame01 = np.multiply(m_frame01,x_bool01)
            dis_frame02 = np.multiply(m_frame02,x_bool02)
            dis_show = np.concatenate((dis_frame01,dis_frame02),axis=1)
        numpixels_frame01,numpixels_frame02 = x_bool01.sum(),x_bool02.sum()
        if numpixels_frame01 > numpixels_frame02:
            numpixels_frame,x_bool,m_frame = numpixels_frame01,x_bool01,m_frame01
        else:
            numpixels_frame,x_bool,m_frame = numpixels_frame02,x_bool02,m_frame02
    else:
        x_bool = ((m_frame-threshold)>=0)
        numpixels_frame = x_bool.sum()
        dis_show = np.multiply(m_frame,x_bool)
    if check_frames:
        plt.imshow(dis_show,cmap='nipy_spectral_r')
        plt.title('Checking frame')
        show_window(noticks=True,winmax=True,closewin=True)

    y_mod = np.multiply(m_frame,x_bool)
    total = y_mod.sum()
    ima = total/numpixels_frame
    return ima
        
            

def load_gridpoints(test):
    cwd = os.getcwd()
    pfilepath = cwd+'\\points_grid\\'+test.filename.replace('.tif','')+'_points_grid.txt'
    ischeck = checkfile(pfilepath,test,checktype=False,isinput=True)
    if ischeck == False:
        return None,None,None,None
    p = np.loadtxt(pfilepath)
    x_left,x_right,y_bot,y_top = int(p[0]),int(p[1]+1),int(p[2]+1),int(p[3])
    return x_left,x_right,y_bot,y_top

def calc_uncertainty(arr,n):
    if n != 10:
        tval = input('What is the t-distribution value you are using?: ')
    else:
        tval = 2.262
    unc = tval*np.std(arr)/np.sqrt(n)
    return unc


def load_areaframe(test):
    ffilepath = os.getcwd().replace('wfire','wfire_cache\\flame_area\\frames\\')+test.filename.replace('.tif','_areaframe_uncropped.npy')
    ischeck = checkfile(ffilepath,test,checktype=False,isinput=True)
    if ischeck == False:
        return None
    frame = np.load(ffilepath)
    fpath = os.getcwd()+'_cache\\flame_area\\frames\\'+test.filename.replace('.tif','_areaframe_ell.npy')
    if os.path.exists(fpath):
        frame = np.load(fpath)
        rows,cols = frame.shape[0],frame.shape[1]
        if rows != cols:
            areas = calc_area(frame,None,threshold,test.spatial_calibration,'frame')
            if areas[0] >= areas[1]:
                start,stop = 0,int(cols/2)
            elif areas[1] > areas[0]:
                start,stop = int(cols/2),cols
            frame = frame[0:rows,start:stop]
    return frame

def calc_saturate(test):
    threshold = get_threshold(test,test.fmc,'other')
    frame = load_areaframe(test)
    if frame is None:
        return 999
    frame_sat = ((frame-255)==0)
    sat = frame_sat.sum()
    frame_act = ((frame-threshold)>=0)
    act = frame_act.sum()
    sat_per = round(sat/act*100,1)
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------')
    print('The percentage of the maximum flame area that is saturated is: ')
    print(sat_per)
    usr = input('Press \'Enter\' to continue, enter \'q\' to stop: ')
    if usr == 'q':
        return 999
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

def plot_height(sets,data,heights,showunc):
    """ Plots average flame heights for different sets of tests
    """
    labels,temperatures,linestyle = get_plotinfo(sets,data) 
    heights_averages = []
    plt.figure(figsize=[9,7.5],dpi=140)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 24})
    plt.tight_layout()
    for i in range(len(heights)):        
        heights_averages.append(np.mean(heights[i]))
        unc,cap = calc_uncertainty(heights[i],10),4
        if showunc == False:
            unc,cap = 0,0
        plt.errorbar(temperatures[i],heights_averages[i],fmt=linestyle[i],yerr=unc,capsize=cap,label=labels[i])
        print('\n',sets[2*i],sets[2*i+1])
        print('Temperature: ',temperatures[i],'height: ',heights_averages[i])
    plt.xlabel('Average exhaust gas temperature ($^{\circ}$C)')
    plt.ylabel('Average flame height (cm)')
    # plt.title('Av')
    plt.legend()
    show_window(noticks=False,winmax=False,closewin=True,showwin=True)
    return

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
    show_window(noticks=False,winmax=False,closewin=True,showwin=True)
    return

def plot_vars(sets,data,vars,var_tags,labels):
    legend_labels,temperatures,linestyle = get_plotinfo(sets,data)
    for i in range(0,len(vars[0])):
        plt.plot(vars[0][i],vars[1][i],linestyle[i],label=legend_labels[i])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()
    show_window(noticks=False,winmax=False,closewin=True,showwin=True)


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

def checkfile(filepath,test,checktype,isinput):
    # Checktype should be true if you are trying to save a file
    # Checktype should be false if you are trying to load a file
    # Function will return True if you are good to go, regardless if saving or loading a file
    if checktype:
        messg = '---------------------------------------------------------\
            \nLooks like there\'s already a file with this name.\
            \nDelete existing file if you are wanting to overwrite\n'
    else:
        messg = '---------------------------------------------------------\
            \nLooks like you are missing the following file.\
            \nPlease go back and generate this file before moving on.\n'
    if os.path.exists(filepath) == checktype:
        # os.system('cls')
        print('\n\n\n\n\n')
        if isinput:
            print(messg)
            print('Filepath: ',filepath,'\nTest number: ',test.testnumber)
            input('Ok (press return)')
        else:
            print('-----------------------------------------------')
            print('Filepath: ',filepath,'\nTest number: ',test.testnumber)
        return False
    else:
        return True

def displayarea(test,cmap_usr):
    threshold = get_threshold(test,test.fmc,'other')
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

def on_release(key):
    print()
    if key == keyboard.Key.enter:
        # Stop listener
        return False

def on_press(key):
    try:
        print()
    except AttributeError:
        print()

def checkframenum(test,cmap_usr,isdisplay):
    frame_num_cropped = load_area(test)[1]
    threshold = get_threshold(test,test.fmc,'other')
    numpixels_filepath = os.getcwd() + '_cache\\numpixels\\' + test.filename.replace('.tif','_numpixels.npy')
    areaframe_numpixels_filepath = os.getcwd() + '_cache\\flame_area\\frames\\' + test.filename.replace('.tif','_areaframe_numpixels.npy')
    areavals_numpixels_filepath = os.getcwd() + '_cache\\flame_area\\vals_numpixels\\' + test.filename.replace('.tif','_areavals_numpixels.npy')
    ischeck = checkfile(numpixels_filepath,test,checktype=False,isinput=False)
    if ischeck == False:
        fname = os.getcwd().replace('wfire','') + test.filename
        numpixels,num_frames,frames = func_wfipa.calc_numpixels(threshold,fname)
        ind = np.where(numpixels==numpixels.max())[0]
        
        img = frames[int(frame_num_cropped)]
        for i in ind:
            img = np.concatenate((img,frames[int(i)]),axis=1)

        np.save(numpixels_filepath,numpixels)
        np.save(areaframe_numpixels_filepath,img)
        np.save(areavals_numpixels_filepath,ind)
        skip = True
    else:
        skip = False
        areaframe_numpixels = np.load(areaframe_numpixels_filepath)
        img = areaframe_numpixels
        ind = np.load(areavals_numpixels_filepath)
    os.system('cls')
    print(test.testnumber)
    print('Max area calculated without a rectangle being removed\n to represent the sample area is at frame number:')
    flame_duration = test.eof - test.ignition_time[1]
    for i in ind:
        print(i)
        print('\nSeconds to max flame area (after ignition): ',round((i-test.ignition_time[1])/(500),2))
        print('\nMaximum flame area occurred ',round((i-test.ignition_time[1])/(flame_duration),2),' of the way through the flaming period\n')
    print('Max area calculated with a rectangle being removed\n to represent the sample area is at frame number:')
    print(int(frame_num_cropped))
    if skip == False:
        check = comp_areavals(test,False)
        if check or isdisplay:
            usr = input('\nPress \'Enter\' to compare frames (or enter \'q\' to continue): ')
            if usr == 'q':
                return
            else:
                comp_frames(img,cmap_usr)
    return

def comp_frames(img,cmap_usr):
    num_rows = img.shape[0]
    calib = np.zeros((num_rows,1))
    calib[0,0] = 255
    img = np.concatenate((img,calib),axis=1)
    img_bool = ((img - 35)>=0)
    img_new = np.multiply(img,img_bool)
    imgshow = np.concatenate((img,img_new),axis=0)
    plt.imshow(imgshow,cmap=cmap_usr)
    plt.title('Left: frame of max area (based on numpixels with cropped rectangle removal). Right: frame of max area (based on numpixels)')
    show_window(noticks=True,winmax=True,closewin=True,showwin=True)
    return

def comp_areavals(test,isprint):
    areaframe_numpixels_filepath = os.getcwd() + '_cache\\flame_area\\frames\\' + test.filename.replace('.tif','_areaframe_numpixels.npy')
    frame = np.load(areaframe_numpixels_filepath)
    num_rows,num_cols = frame.shape[0],frame.shape[1]
    if num_cols > 512:
        num_cols = 512
    cols = int(num_cols/2)
    frame_01 = frame[0:num_rows,0:cols]
    frame_02 = frame[0:num_rows,cols:num_cols]
    threshold = get_threshold(test,test.fmc,'other')
    area_01 = calc_area(frame_01,None,threshold,test.spatial_calibration,'frame')
    area_02 = calc_area(frame_02,None,threshold,test.spatial_calibration,'frame')
    per_diff = np.abs((area_01-area_02)/area_01)*100
    if isprint:
        print('\nTest number: ',test.testnumber)
        print('Area01: ',round(area_01,4),' cm^2')
        print('Area02: ',round(area_02,4),' cm^2')
        print('Percent difference: ',round(per_diff,2),'\n')
        fpath = os.getcwd()+'_cache\\flame_area\\frames\\'+test.filename.replace('.tif','_areaframe_ell.npy')
        if os.path.exists(fpath):
            img = np.load(fpath)
            areas = calc_area(img,None,threshold,test.spatial_calibration,'frame')
            if isinstance(areas,list):
                area_01,area_02 = areas[0],areas[1]
            else:
                area_02 = areas
            per_diff = np.abs((area_01-area_02)/area_01)*100
            print('----Updated area----')
            print('Test number: ',test.testnumber)
            print('Area01_original: ',round(area_01,4),' cm^2')
            print('Area01_updated: ',round(area_02,4),' cm^2')
    if per_diff >= 10:
        if isprint == False:
            print('\nTest number: ',test.testnumber)
            print('Area01: ',round(area_01,4),' cm^2')
            print('Area02: ',round(area_02,4),' cm^2')
            print('Percent difference: ',round(per_diff,2),'\n')

            fpath = os.getcwd()+'_cache\\flame_area\\frames\\'+test.filename.replace('.tif','_areaframe_ell.npy')
            if os.path.exists(fpath):
                img = np.load(fpath)
                areas = calc_area(img,None,threshold,test.spatial_calibration,'frame')
                per_diff = np.abs((areas[0]-areas[1])/areas[0])*100
                print('----Updated area----')
                print('Test number: ',test.testnumber)
                print('Area01: ',round(areas[0],4),' cm^2')
                print('Area02: ',round(areas[1],4),' cm^2')
                print('Percent difference: ',round(per_diff,2),'\n---------')
        return True
    return False


def change_cmap(cmap):
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------','\n')
    print('Current colormap is: ',cmap)
    print('Choose one of the following maps (hit enter for no change)')
    print(' 1 - viridis')
    print(' 2 - twilight')
    print(' 3 - turbo')
    print(' 4 - CMRmap')
    print(' 5 - flag')
    print(' 6 - gist_stern_r')
    print(' 7 - nipy_spectral_r')
    print(' 8 - tab20')
    print(' 9 - Set3')
    print(' 10 - greys')
    print(' 11 - nipy_spectral')
    print(' 12 - gist_earth_r')
    usr = input('Selected option: ')
    if usr == '':
        return cmap
    usr = int(usr)
    if usr < 1 or usr > 12:
        return cmap
    cmaps = ['viridis','twilight','turbo','CMRmap','flag','gist_stern_r','nipy_spectral_r','tab20','Set3','Greys','nipy_spectral','gist_earth_r']
    return cmaps[usr-1]

def plot_numpixelsarea(test,showmax,threshold):
    usr = input('Continue? (b)')
    if usr == 'b':
        return 999
    if threshold == 35:
        numpixels_filepath = os.getcwd() + '_cache\\numpixels\\' + test.filename.replace('.tif','_numpixels.npy')
    else:
        numpixels_filepath = os.getcwd() + '_cache\\numpixels\\' + test.filename.replace('.tif','_numpixels_'+str(threshold)+'.npy')
    areavals_numpixels_filepath = os.getcwd() + '_cache\\flame_area\\vals_numpixels\\' + test.filename.replace('.tif','_areavals_numpixels.npy')
    # msg = 'You need to run \'Check frame number\' to generate this file'
    ischeck = checkfile(numpixels_filepath,test,checktype=False,isinput=False)
    if ischeck == False:
        print('No values have been saved for this file & threshold value. Loading frames......\n')
        fname = os.getcwd().replace('wfire','') + test.filename
        numpixels = func_wfipa.calc_numpixels(threshold,fname)[0]
        if threshold == 35:
            usr = 'y'
        else:
            usr = input('Would you like to save the data for this threshold value? (y/n)')
        if usr == 'y':
            np.save(numpixels_filepath,numpixels)
    else:
        numpixels = np.load(numpixels_filepath)
    ignition_frame = test.ignition_time[1]
    if ignition_frame != 0:
        z_indices = np.where(numpixels[0:ignition_frame]==0)
        if z_indices[0].shape[0] == 0:
            last_z = 0
        else:
            last_z = z_indices[0][-1]
        print(last_z)
        # last_z/=500
        print(last_z)
    num_frames = len(numpixels)
    x,xlabel = np.linspace(1,num_frames,num_frames),'frame number'
    # if showmax == False:
    #     x/=500
    #     xlabel = 'time (s)'
    pixel_length = test.spatial_calibration*100 # change from m to cm
    if pixel_length == 0:
        input('There is no spatial calibration for this test so the default value of 0.0004 m/pixel is being used (hit enter to continue)')
        pixel_length = 0.0004*100
    pixel_area = pixel_length**2
    areapixels = numpixels*pixel_area # cm^2
    areapixels_mod = areapixels[0:ignition_frame]
    x_mod = x[0:ignition_frame]
    max_area = areapixels.max()
    # max_area = areapixels_mod.max()
    plt.figure(figsize=[10,7],dpi=140)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 24})
    plt.tight_layout()
    plt.plot(x,areapixels,linewidth=0.5)
    # plt.plot(x_mod,areapixels_mod,linewidth=0.5)
    if ignition_frame != 0:
        plt.plot([last_z,last_z],[0,max_area],':')
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
    mpl.cursor()
    show_window(noticks=False,winmax=False,closewin=False,showwin=True)
    return

def change_errbar(showunc):
    if showunc:
        switch = 'on'
        msg = 'Would you like to turn errorbars off? (y/n): '
    else:
        switch = 'off'
        msg = 'Would you like to turn errorbars on? (y/n): '
    print('Currently errorbars are switched ',switch,' for plots')
    usr = input(msg)
    if usr == 'y' and switch == 'off':
        showunc = 1
    elif usr == 'y' and switch == 'on':
        showunc = 0
    elif usr == 'n' and switch == 'off':
        showunc = 0
    elif usr == 'n' and switch == 'on':
        showunc = 1
    else:
        input('Error')
    return showunc

def selectarea(test):
    areaframe_uncropped_filepath = os.getcwd() + '_cache\\flame_area\\frames\\' + test.filename.replace('.tif','_areaframe_uncropped.npy')
    img = np.load(areaframe_uncropped_filepath)
    ell_filepath = os.getcwd() + '_cache\\flame_area\\ell\\' + test.filename.replace('.tif','_ell.npy')
    areaframe_ell_path = os.getcwd()+'_cache\\flame_area\\frames\\'+test.filename.replace('.tif','_areaframe_ell.npy')
    compare,initial = comp_areavals(test,False),True
    if os.path.exists(areaframe_ell_path) == True:
        img,initial = np.load(areaframe_ell_path),False
    if compare == True and initial == True:
        fpath = os.getcwd() + '_cache\\flame_area\\frames\\' + test.filename.replace('.tif','_areaframe_numpixels.npy')
        img = np.load(fpath)

    threshold = get_threshold(test,test.fmc,'area')
    num_rows = img.shape[0]
    calib = np.zeros((num_rows,1))
    calib[0,0] = 255
    dis_img = np.concatenate((img,calib),axis=1)
    dis_bool = ((dis_img - threshold)>=0)
    dis_new = np.multiply(dis_img,dis_bool)

    running,count = True,0
    while running:
        p = get_points(dis_new,test,'selectarea')[0]
        pfilepath = os.getcwd() + '_cache\\flame_area\\selectarea\\'+test.filename.replace('.tif','_selectpoints.npy')
        left,right,bottom,top=p[0][0],p[0][1],p[1][0],p[1][1]
        center = [(left+right)/2,(bottom+top)/2]
        w = right-left
        h = bottom-top
        ell_info = np.array((center,[w,h]))
        if os.path.exists(ell_filepath) == True:
            ell_list = np.load(ell_filepath)
            ell_list = np.append(ell_list,ell_info,axis=0)
        else:
            if count == 0:
                ell_list = ell_info
            elif count != 0:
                ell_list = np.append(ell_list,ell_info,axis=0)

        for i in range(int(ell_list.shape[0]/2)):
            center = ell_list[2*i]
            w = ell_list[2*i+1][0]
            h = ell_list[2*i+1][1]
            ell = pat.Ellipse(center,w,h,edgecolor='black',facecolor='none')
            img_new = edit_frame(img,ell,[center,w,h])
    
        #------ 
        threshold = get_threshold(test,test.fmc,'area')
        num_rows = img_new.shape[0]
        calib = np.zeros((num_rows,1))
        calib[0,0] = 255
        dis_img = np.concatenate((img_new,calib),axis=1)
        dis_bool = ((dis_img - threshold)>=0)
        dis_new = np.multiply(dis_img,dis_bool)
        imgshow = np.concatenate((img_new,dis_new),axis=1)
        #--------

        plt.imshow(imgshow,cmap='nipy_spectral_r')
        show_window(noticks=True,winmax=True,closewin=True)
        count+=1
        print(test.testnumber)
        print()
        usr = input(' Would you like to continue selecting areas? (y/n/q) \n(If you would like to adjust the threshold, input \'a\' instead): ')
        if usr == 'n':
            running = False
        elif usr == 'y':
            img = img_new
        elif usr == 'a':
            change_threshold(test)         
        elif usr == 'q':
            np.save(ell_filepath,ell_list)
            np.save(areaframe_ell_path,img_new)
            return 999
    np.save(ell_filepath,ell_list)
    np.save(areaframe_ell_path,img_new)
    return

def change_threshold(test):
    ref_num = int(test.testnumber-1)
    fpath = os.getcwd()+'_cache\\adjusted_thresholds.npy'
    thresholds = np.load(fpath)
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------')
    print('The current threshold is ',thresholds[ref_num])
    usr = input('Would you like to add or subtract from the threshold? (a/s)')
    if usr == 'a':
        msg,mult = 'add to',1
    elif usr == 's':
        msg,mult = 'subtract from',-1
    else:
        input('Error')
        return
    msg = 'How much would you like to '+msg+' the threshold?: '
    adj = int(input(msg))
    input(adj)
    thresholds[ref_num] = thresholds[ref_num] + adj*mult
    input(thresholds[ref_num])
    np.save(fpath,thresholds)
    return

def edit_frame(img_new,ell,inf):
    c,w,h = inf[0],inf[1],inf[2]
    x = int(c[0]-w/2)
    y = int(c[1]-h/2)
    # print(x,y)
    for i in range(int(w)):
        for j in range(int(h)):
            ispoint = ell.contains_point([x+i,y+j])
            if ispoint:
                img_new[y+j][x+i]=0
    return img_new

def run_centerpoints(test):
    fname = os.getcwd().replace('wfire','') + test.filename
    threshold = get_threshold(test,test.fmc,tag='other')
    numpixels,num_frames,frames = func_wfipa.calc_numpixels(threshold,fname)
    ignition_time = test.ignition_time[1]
    eof = test.eof
    num_rows,num_cols = 8,8
    centerpoints = np.zeros((num_rows,num_cols,int(eof-ignition_time)))
    # centerpoints = np.zeros((num_rows,num_cols,int(num_frames-eof)-1))
    # change = np.zeros((num_rows,num_cols,int(num_frames-eof)-1))
    change = np.zeros((num_rows,num_cols,int(eof-ignition_time)))
    start = int(ignition_time)
    stop = int(ignition_time + centerpoints.shape[2])
    # start = int(eof)
    # stop = int(eof+centerpoints.shape[2])
    frame_ind,nancount = 0,0

    for i in range(start,stop):
        ref_frame = frames[i].astype(float)
        sec_num = 0
        for j in range(num_rows):
            for k in range(num_cols):
                sec_num+=1
                row_slice = [int(j*256/num_rows),int((j+1)*(256/num_rows))]
                col_slice = [int(k*256/num_cols),int((k+1)*(256/num_cols))]
                section = ref_frame[row_slice[0]:row_slice[1],col_slice[0]:col_slice[1]]
                if i == start:
                    prev_val = np.sqrt(16**2+16**2)
                else:
                    prev_val = centerpoints[j,k,frame_ind-1]
                centerpoint = calc_centerpoint(section,threshold,prev_val,sec_num)
                if centerpoint == 0:
                    for ii in section:
                        print(ii)
                    print('section number: ',sec_num)
                    plt.imshow(section)
                    show_window(noticks=True,winmax=False,closewin=True,showwin=True)
                if math.isnan(centerpoint):
                    nancount+=1
                change[j,k,frame_ind] = abs((centerpoint-prev_val)/prev_val)
                centerpoints[j,k,frame_ind] = centerpoint
        frame_ind+=1
    print('The nancount is: ',nancount)
    sec_num = 0
    for i in range(num_rows):
        for j in range(num_cols):
            points_section = centerpoints[i,j,:]
            change_section = change[i,j,:]
            sec_num+=1
            print('Section: ',sec_num, change_section.max())
            if change_section.max() == 0:
                continue
            usr = input('Press enter to move on')
            if usr == 'b':
                return 999
            x = np.linspace(0,len(points_section),len(points_section))
            plt.figure()
            plt.plot(x,points_section,linewidth=0.5)
            plt.ylim(0,45)
            show_window(noticks=False,winmax=False,closewin=False,showwin=False)
            plt.figure()
            plt.plot(x,change[i,j,:],linewidth=0.5)
            plt.ylim(0,1)
            show_window(noticks=False,winmax=False,closewin=True,showwin=True)
    return

def calc_centerpoint(section,threshold,prev_val,sec_num):
    if section.max() < threshold:
        return prev_val
    bool_frame = ((section-threshold)>=0)
    inds = np.where(bool_frame==True)
    xavg = np.average(inds[0])
    yavg = np.average(inds[1])
    mag = np.sqrt(xavg**2+yavg**2)
    if mag == 0:
        mag = 1
    return mag

def save_burnout(test):
    fpath = os.getcwd() + '_cache\\burnout\\' + test.filename.replace('.tif','_burnoutframes.npy')
    ischeck = checkfile(fpath,test,checktype=True,isinput=True)
    if ischeck is False:
        return 999
    test_filepath = os.getcwd().replace('wfire','') + test.filename
    img,filename = readfile(test_filepath,True)
    frames,num_frames,num_rows,num_cols = get_image_properties(img)
    burnout_img = frames[int(test.eof-1)]
    thresh = get_threshold(test,test.fmc,tag='other')
    img = burnout_img.astype(float)
    arr = (img-thresh)>0
    burnout_img_mod = np.multiply(burnout_img,arr)
    burnout_imgs = np.concatenate((burnout_img,burnout_img_mod),axis=1)
    np.save(fpath,burnout_imgs)
    return True

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

def load_flameheight_points(test,heatmap):
    cwd = os.getcwd()
    pfilepath = cwd+'_cache\\points\\flame_height\\'+test.filename.replace('.tif','')+'_points_height.txt'
    ischeck = checkfile(pfilepath,test,False,False)
    if ischeck == False:
        input('First point selected will be the datum. Then select any number of points to profile off flaming vs smoldering area')
        points,num_points = get_points(heatmap,test,points_type='flameheight')
        save_points(test,points,num_points,'flame_height')
    else:
        p = np.loadtxt(pfilepath,unpack=True)
        points = []
        for i in range(int(len(p)/2)):
            points.append([p[i*2],p[i*2+1]])
        num_points = len(points)
    datum,values = points[0],[]
    for i in range(num_points-1):
        values.append(points[i+1])
    slope01 = (values[0][1]-values[1][1])/(values[0][0]-values[1][0])
    b01 = values[0][1]-slope01*values[0][0]
    if b01 > datum[1]:
        b01 = datum[1]
    values[0] = [0,b01]
    slope02 = (values[-1][1]-values[-2][1])/(values[-1][0]-values[-2][0])
    b02 = values[-1][1]-slope02*values[-1][0]
    num_cols = heatmap.shape[1]
    values[-1] = [num_cols,slope02*num_cols+b02]
    if values[-1][1] > datum[1]:
        values[-1][1] = datum[1]
    xvals,yvals = [],[]
    for i in range(len(values)):
        xvals.append(values[i][0])
        yvals.append(values[i][1])
    return datum,xvals,yvals

def get_crop(test,heatmap):
    input('Select 4 points per region you want cropped out (cropped regions will be rectangles). First two points will determine width, second two poitns will determine height')
    points,num_points = get_points(heatmap,test,points_type='crop')
    save_points(test,points,num_points,points_type='crop')
    return points,num_points

def load_crop(test,heatmap,ischeck,pfilepath):
    if ischeck == False:
        points,num_points = get_crop(test,heatmap)
    else:
        p = np.loadtxt(pfilepath,unpack=True)
        points = []
        for i in range(int(len(p)/2)):
            points.append([p[i*2],p[i*2+1]])
        num_points = len(points)
    return points,num_points

def create_mask(num_rows,num_cols,points,num_rec):
    mask = np.zeros((num_rows,num_cols))+1
    for i in range(num_rec):
        w_vals = [points[i*4][0],points[i*4+1][0]]
        h_vals = [points[i*4+2][1],points[i*4+3][1]]
        top_row,bottom_row = int(min(h_vals)),int(max(h_vals))
        left_col,right_col = int(min(w_vals)),int(max(w_vals))
        if right_col > 0.9*num_cols:
            right_col = num_cols
        if top_row < 15:
            top_row = 0
        width,height = right_col-left_col,bottom_row-top_row
        # print(width,height)
        for j in range(height):
            for k in range(width):
                # print(j,k,'\n',left_col+k)
                mask[top_row+j,left_col+k] = 0
    return mask

def find_flame_height(test,args):
    heatmap = load_heatmap(test,args[1])
    datum,xvals,yvals = load_flameheight_points(test,heatmap)

    # plt.plot(xvals,yvals)
    # plt.imshow(heatmap,cmap=args[0])
    # show_window(noticks=False,winmax=False,closewin=True,showwin=True)


    fname = os.getcwd().replace('wfire','') + test.filename
    threshold = get_threshold(test,test.fmc,tag='other')
    numpixels,num_frames,frames = func_wfipa.calc_numpixels(threshold,fname)
    num_rows,num_cols = frames[0].shape[0],frames[1].shape[1]
    flaming_frames = 0
    row_heights = []

    ans = input('Would you like to crop regions in the frame? (y/n)')
    if ans == 'y':
        cwd = os.getcwd()
        pfilepath = cwd+'_cache\\points\\crop\\'+test.filename.replace('.tif','')+'_points_crop.txt'
        ischeck = checkfile(pfilepath,test,False,False)
        points,num_points = load_crop(test,heatmap,ischeck,pfilepath)
        num_rec = int(num_points/4)
        mask = create_mask(num_rows,num_cols,points,num_rec)

    heatmap_crop = np.multiply(heatmap,mask)
    plt.imshow(heatmap_crop,cmap=args[0])
    show_window(noticks=True,winmax=True,closewin=True,showwin=True)

    lines_peak_row = int(min(yvals))
    lines_bottom_row = int(max(yvals))
    k_vals = []
    for i in range(num_frames):
        ref_frame = frames[i].astype(float)
        if ans == 'y':
            ref_frame = np.multiply(ref_frame,mask)
        recorded,change = False,False
        for j in range(lines_bottom_row):
            if ref_frame[j].max() >= threshold:
                roi = j # row of interest
                if j < lines_peak_row:
                    # heatmap[j] = 0 ###
                    flaming_frames+=1
                    if recorded:
                        print('Recording more than one height for this frame #: ',i)
                        input('Press \'Enter\', function will exit')
                        return
                    row_heights.append(roi)
                    recorded = True
                    break
                elif j >= lines_peak_row:
                    for k in range(num_cols):
                        k_vals.append(k)
                        pix = ref_frame[j,k]
                        if pix < threshold:
                            continue
                        for ii in range(len(xvals)):
                            if ii != len(xvals)-1:
                                if xvals[ii] == xvals[ii+1]:
                                    xvals[ii+1]*=1.01
                                    change = True
                            if k >= xvals[ii] and k <= xvals[ii+1]:
                                x1,y1 = xvals[ii],yvals[ii]
                                x2,y2 = xvals[ii+1],yvals[ii+1]
                                if change:
                                    change = False
                                    xvals[ii+1]/=1.01
                                break
                            else:
                                if change:
                                    change = False
                                    xvals[ii+1]/=1.01
                        m = (y2-y1)/(x2-x1)
                        b = y2-m*x2
                        xcheck,ycheck = k,j
                        yactual = m*xcheck+b
                        if ycheck > yactual:
                            continue
                        else:
                            flaming_frames+=1
                            row_heights.append(roi)
                            recorded = True
                            # heatmap[j,k] = heatmap.max() ###
                            break
                    break
    print('\nNumber of frames with flame detected: ',flaming_frames)
    print('Flaming duration: ',test.eof-test.ignition_time[1])
    duration = test.eof - test.ignition_time[1]
    if flaming_frames > duration+1:
        print('Test number: ',test.testnumber)
        print('ERROR, flaming in this function is detected longer than flaming duration. Need to redo points selection')
        usr = input('Continue or return? (c/r)')
        if usr == 'r':
            return
    # input('Press enter')
    ######
    current_row = min(row_heights)
    running = True
    percentage = None
    while running:
        old_percentage = percentage
        count = 0
        for i in row_heights:
            if i <= current_row:
                count+=1
        percentage = round(count/flaming_frames,1)
        if percentage >= 0.5:
            avg_flame_height = current_row
            running = False
            print('Row of average flame height: ',avg_flame_height,'\nRow representing the bottom of the lines, or bottom of flame zone: ',lines_bottom_row,'\nPercentage: ',percentage,'\nPrevious percentage: ',old_percentage)
        else:
            current_row+=1
            if current_row >= lines_bottom_row:
                print('Error: The row being checked for average flame height has now reached the bottom of the upper flame zone defined by user selected points.')
                print('Current_row: ',current_row,'\nlines_bottom_row: ',lines_bottom_row,'\nPercentage: ',percentage,'\nPrevious percentage: ',old_percentage)
                input('Hit \'Enter\' to return')
                return
                
    heatmap_crop[avg_flame_height] = heatmap.max()
    heatmap_crop[int(datum[1])] = heatmap.max()
    heatmap_crop[lines_peak_row] = heatmap.max()
    spatial_calib = test.spatial_calibration*100 # multiplying by 100 makes it cm/pix
    avg_flame_height = (datum[1]-avg_flame_height)*spatial_calib
    print(round(avg_flame_height,2),' cm')
    displaymaps(heatmap_crop,'all',cmap_usr='nipy_spectral_r',xvals=xvals,yvals=yvals)
    usr = input('Continue or go back? (enter/b)')
    if usr == 'b':
        return 999