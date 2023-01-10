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
import cv2 as cv
import keyboard
import matplotlib.animation as animation
import func_wfipa
import matplotlib.patches as pat
from pynput import keyboard

def importdata(datafile='data.txt',namesfile='filenames.txt'):
    """
    Loads data from txt files, uses data to create a list of Test objects
    and returns this list as a tuple
    """
    month,day,year,testnum,set_,orientation,height,temp,fmc,time,frame,spatial,eof = np.loadtxt(datafile,unpack=True)
    filenames = np.loadtxt(namesfile,dtype=str)
    data = []
    for i in range(0,len(filenames)):
        filename = filenames[i]+'.tif'
        if set_[i] == 0 or set_[i] == 1:
            stype = 'individual'
        elif set_[i] == 2:
            stype = 'multiple'
        else:
            stype = None
        ig_test = Test(filename,(month[i],day[i],year[i]),testnum[i],(stype,orientation[i],height[i],temp[i]),fmc[i],(time[i],int(frame[i])),spatial[i],eof[i])
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
        show_window(noticks=True,winmax=True)

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

def get_heatmaps(test,save,thresh):
    """.....
    """
    print(test.testnumber)
    name = test.filename.replace('.tif','')
    cwd = os.getcwd()
    cwd = cwd+'\\heatmaps\\'+name+'_heatmap.npy'
    ischeck = checkfile(cwd,test,checktype=True,isinput=True)
    if ischeck == False:
        return
    filepath = os.getcwd().replace('wfire','') + test.filename
    img,filename = readfile(filepath,True)
    frames,num_frames,num_rows,num_cols = get_image_properties(img)
    heatmap = np.zeros((num_rows,num_cols))
    frame_num = 0
    print('threshold value: ',thresh)
    for j in frames:
        j = j.astype(float)
        arr = (j-thresh)>0
        heatmap+=arr
        frame_num += 1
        if frame_num > test.eof:
            break
    if save is True:
        np.save(cwd,heatmap)
    return None

def displaymaps(heatmap):
    """...
    """
    usr = input('Continue (\'b\' to go back)')
    if usr == 'b':
        return 999
    if heatmap is None:
        input('Heatmap has not been loaded, most likely because there is no file saved for it (Hit \'Enter\')')
    num_rows = heatmap.shape[0]
    calib = np.zeros((num_rows,1))
    calib[0,0] = 4500
    imgs = np.concatenate((heatmap,calib),axis=1)
    plt.imshow(imgs,cmap='nipy_spectral_r')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    show_window(noticks=True,winmax=True)
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
    ischeck = checkfile(pfilepath,test,checktype=True,isinput=True)
    if ischeck == False:
        return 0,0,False
    num_rows = img.shape[0]
    calib = np.zeros((num_rows,1))
    calib[0,0] = 4500
    if points_type != 'selectarea':
        img = np.concatenate((img,calib),axis=1)
    plt.imshow(img)
    plt.get_current_fig_manager().window.showMaximized()
    if points_type == 'grid' or points_type == 'selectarea':
        p = plt.ginput(n=-1,timeout=-1,show_clicks=True)
        p = refine_gridpoints(p)
    else:
        p = plt.ginput(num_points)
##    print(p[0][0],p[0][1])
    plt.close()

    return p,num_points

def refine_gridpoints(p):
    x_left,x_right,y_bot,y_top = 255,0,0,255
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
    is_1,is_2,is_3,is_4,is_5,is_6=0,0,0,0,0,0
    while running:
        os.system('cls')
        print('\n\n\n\n\n','--------------------------------------------------','\n')
        print('Individual (65mm): 1')
        print('Individual (83mm): 2')
        print('Multiple-Live (65mm): 3')
        print('Multiple-Live (83mm): 4')
        print('Multiple-dried (65mm): 5')
        print('Multiple-dried (83mm): 6')
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
        elif usr == 'f':
            running = False
    sets_new = []
    for i in range(len(sets)):
        var = sets[i]
        for j in range(len(var)):
            sets_new.append(var[j])
    return sets_new

def load_heatmap(test):
    """...
    """
    pathname = test.filename.replace('.tif','')
    cwd = os.getcwd()
    path = cwd+'\\heatmaps\\'+pathname+'_heatmap.npy'
    msg = 'You can create this file by generating a heatmap'
    ischeck = checkfile(path,test,checktype=False,isinput=True)
    if ischeck == False:
        return None
    heatmap = np.load(path)
    return heatmap


def save_points(test,p,num_points,points_type):
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
    else:
        input('Error (press \'Enter\' to return')
        return
    ischeck = checkfile(pfilepath,test,checktype=True,isinput=True)
    if ischeck == False:
        return
    pfile = open(pfilepath,'x')
    if points_type == 'profile':
        pfile.write(str(p[0][0])+' '+str(p[0][1])+' '+str(p[1][0])+' '+str(p[1][1]))
    elif points_type == 'timeline' or points_type == 'vert' or points_type == 'grid':
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
    show_window(noticks=True,winmax=False)

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
    show_window(noticks=False,winmax=False)

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
    show_window(noticks=False,winmax=False)
        
def change_ylim(ylim):
    ylim = input('Set ylim for plotting: ')
    return ylim

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
    heatmap = load_heatmap(test)
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
    show_window(noticks=True,winmax=False)
            
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
    show_window(noticks=False,winmax=False)
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
    if fmc == 0:
        threshold = 50
    else:
        threshold = 35

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
        threshold = get_threshold(test.fmc)
        max_flamearea = calc_area(None,areaframe_ell_path,threshold,test.spatial_calibration,'file')
    else:
        print('\nTest number ',test.testnumber,' is using an outdated area')
    return max_flamearea,frame_num     

def calc_area(ref_frame,filename,threshold,pixel_length,tag):      
    if tag == 'file':
        ref_frame = np.load(filename)
    ref_frame = ref_frame.astype(float)
    x_bool = ((ref_frame-threshold)>=0)
    numpixels_frame = x_bool.sum()
    pixel_area = pixel_length**2
    flame_area = pixel_area*numpixels_frame*(100**2)
    return flame_area

def get_threshold(fmc):
    if fmc == 0:
        threshold = 50
    else:
        threshold = 35
    return threshold

def get_ima(test):
    mfilepath = os.getcwd()+'_cache\\flame_area\\frames\\'+test.filename.replace('.tif','_areaframe_cropped.npy')
    ischeck = checkfile(mfilepath,test,checktype=False,isinput=True)
    if ischeck == False:
        return 999
    m_frame = np.load(mfilepath)
    fmc = test.fmc
    if fmc == 0:
        threshold = 50
    else:
        threshold = 35
    x_bool = ((m_frame-threshold)>=0)
    numpixels_frame = x_bool.sum()
    y_mod = np.multiply(m_frame,x_bool)
    total = y_mod.sum()
    ima = total/numpixels_frame
    # print('Average intensity of maximmum flame area: ',ima)
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
    return frame

def calc_saturate(test):
    if test.fmc == 0:
        threshold = 50
    else:
        threshold = 35
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
    for i in range(len(max_flamearea_sets)):           
        area_averages.append(np.mean(max_flamearea_sets[i]))
        unc,cap = calc_uncertainty(max_flamearea_sets[i],10),4
        if showunc == False:
            unc,cap = 0,0
        plt.errorbar(temperatures[i],area_averages[i],fmt=linestyle[i],yerr=unc,capsize=cap,label=labels[i])
    plt.xlabel('Average exhaust gas temperature $^{\circ}C$')
    plt.ylabel('Maximum flame area (cm$^2$)')
    plt.title('Average maximum flame area')
    plt.legend()
    show_window(noticks=False,winmax=False)
    return

def plot_ima(sets,data,ima_sets,showunc):
    labels,temperatures,linestyle = get_plotinfo(sets,data) 
    ima_averages = []
    for i in range(len(ima_sets)):        
        ima_averages.append(np.mean(ima_sets[i]))
        unc,cap = calc_uncertainty(ima_sets[i],10),4
        if showunc == False:
            unc,cap = 0,0
        plt.errorbar(temperatures[i],ima_averages[i]/255,fmt=linestyle[i],yerr=unc/255,capsize=cap,label=labels[i])
    plt.xlabel('Average exhaust gas temperature $^{\circ}C$')
    plt.ylabel('Average intensity of max flame area')
    plt.title('Average normalized light intensity of maximum flame area')
    plt.ylim(0,1)
    plt.legend()
    show_window(noticks=False,winmax=False)
    return

def plot_igtime(sets,data,igtimes,showunc):
    """ Plots ignition times for different sets of tests
    """
    labels,temperatures,linestyle = get_plotinfo(sets,data) 
    igtimes_averages = []
    for i in range(len(igtimes)):        
        igtimes_averages.append(np.mean(igtimes[i]))
        unc,cap = calc_uncertainty(igtimes[i],10),4
        if showunc == False:
            unc,cap = 0,0
        plt.errorbar(temperatures[i],igtimes_averages[i]/500,fmt=linestyle[i],yerr=unc/500,capsize=cap,label=labels[i])
    plt.xlabel('Average exhaust gas temperature $^{\circ}C$')
    plt.ylabel('Average ignition time (s)')
    plt.title('Average ignition times')
    plt.legend()
    show_window(noticks=False,winmax=False)
    return

def plot_dur(sets,data,dur,showunc):
    labels,temperatures,linestyle = get_plotinfo(sets,data) 
    dur_averages = []
    for i in range(len(dur)):        
        dur_averages.append(np.mean(dur[i]))
        unc,cap = calc_uncertainty(dur[i],10),4
        if showunc == False:
            unc,cap = 0,0
        plt.errorbar(temperatures[i],dur_averages[i]/500,fmt=linestyle[i],yerr=unc/500,capsize=cap,label=labels[i])
    plt.xlabel('Average exhaust gas temperature $^{\circ}C$')
    plt.ylabel('Average flaming duration (s)')
    plt.title('Average flaming duration')
    plt.legend()
    show_window(noticks=False,winmax=False)
    return

def print_igtimes_avg(sets,data,igtimes):
    os.system('cls')
    print('\n\n\n\n\n','--------------------------------------------------','\n')
    igtimes_averages = []
    temperatures = get_plotinfo(sets,data)[1]
    for i in range(len(igtimes)):
        igtimes_averages.append(np.mean(igtimes[i]))
        unc = calc_uncertainty(igtimes[i],10)
        start,stop=int(sets[i]),int(sets[i+1])
        print('Temperature: ',temperatures[i])
        # print('Test numbers: ',start,'-',stop)
        print('Average ignition time: ',round(igtimes_averages[i]/500,2))
        print('Uncertainty: ',round(unc/500,2),'\n\n')
    input('Press enter to continue')
    return


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
        elif test.fmc != 0 and test.set_type[0] == 'multiple':
            linestyle.append('g^')
            if labellive_m == True:
                labels.append(None)
            elif labellive_m == False:
                labels.append('Live fuel-'+test.set_type[0])
                labellive_m = True
        elif test.fmc != 0 and test.set_type[0] == 'individual':
            linestyle.append('bo')
            if labellive_i == True:
                labels.append(None)
            elif labellive_i == False:
                labels.append('Live fuel-'+test.set_type[0])
                labellive_i = True
    return labels,temperatures,linestyle

def checkfile(filepath,test,checktype,isinput):
    if checktype:
        messg = '---------------------------------------------------------\
            \nLooks like there\'s already a file with this name.\
            \nDelete existing file if you are wanting to overwrite\n'
    else:
        messg = '---------------------------------------------------------\
            \nLooks like you are missing the following file.\
            \nPlease go back and generate this file before moving on.\n'
    if os.path.exists(filepath) == checktype:
        os.system('cls')
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
    return

def displayarea(test,cmap_usr):
    show = False
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
        areaframe_ell = np.concatenate((areaframe_ell,calib),axis=1)
        frame_bool = ((areaframe_ell-threshold)>=0)
        img_add = np.multiply(areaframe_ell,frame_bool)
        img_new = np.concatenate((areaframe_ell,img_add),axis=0)
        imgshow = np.concatenate((imgshow,img_new),axis=1)
    if show:
        os.system('cls')
        print('\n\n\n\n\n','--------------------------------------------------')
        usr = input('Press \'Enter\' to continue, enter \'q\' to stop: ')
        if usr == 'q':
            return 999
        plt.imshow(imgshow,cmap=cmap_usr)
        title = 'Test number: '+str(int(test.testnumber))
        plt.title(title)
        show_window(noticks=True,winmax=True)
    return

def show_window(noticks,winmax):
    if noticks:
        plt.tick_params(axis='both',bottom=False,labelbottom=False,left=False,labelleft=False)
    if winmax:
        plt.get_current_fig_manager().window.showMaximized()
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
    plt.close()
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

def checkframenum(test,cmap_usr):
    frame_num_cropped = load_area(test)[1]
    if test.fmc == 0:
        threshold = 50
    else:
        threshold = 35
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
    for i in ind:
        print(i)
    print('Max area calculated with a rectangle being removed\n to represent the sample area is at frame number:')
    print(int(frame_num_cropped))
    # skip = True
    if skip == False:
        check = comp_areavals(test)
        if check:
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
    show_window(noticks=True,winmax=True)
    return

def comp_areavals(test):
    areaframe_numpixels_filepath = os.getcwd() + '_cache\\flame_area\\frames\\' + test.filename.replace('.tif','_areaframe_numpixels.npy')
    frame = np.load(areaframe_numpixels_filepath)
    num_rows,num_cols = frame.shape[0],frame.shape[1]
    if num_cols > 512:
        num_cols = 512
    cols = int(num_cols/2)
    frame_01 = frame[0:num_rows,0:cols]
    frame_02 = frame[0:num_rows,cols:num_cols]
    if test.fmc != 0:
        threshold = 35
    elif test.fmc == 0:
        threshold = 50
    area_01 = calc_area(frame_01,None,threshold,test.spatial_calibration,'frame')
    area_02 = calc_area(frame_02,None,threshold,test.spatial_calibration,'frame')
    per_diff = np.abs((area_01-area_02)/area_01)*100
    if per_diff >= 10:
        print('Test number: ',test.testnumber)
        print('Area01: ',round(area_01,4),' cm^2')
        print('Area02: ',round(area_02,4),' cm^2')
        print('Percent difference: ',round(per_diff,2),'\n')
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
    print(' 6 - gist_ncar')
    print(' 7 - nipy_spectral_r')
    print(' 8 - tab20')
    print(' 9 - Set3')
    print(' 10 - turbo')
    usr = input('Selected option: ')
    if usr == '':
        return cmap
    usr = int(usr)
    if usr < 1 or usr > 10:
        return cmap
    cmaps = ['viridis','twilight','turbo','CMRmap','flag','gist_ncar','nipy_spectral_r','tab20','Set3','turbo']
    return cmaps[usr-1]

def plot_numpixelsarea(test,showmax):
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

    plt.plot(x,areapixels,linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel('pixel area (cm$^2$)')
    if showmax:
        frame_num_cropped = load_area(test)[1]
        frame_num_numpixels = np.load(areavals_numpixels_filepath)
        x_c,y = [frame_num_cropped,frame_num_cropped],[0,np.amax(areapixels)/2]
        x_n = [frame_num_numpixels,frame_num_numpixels]
        plt.plot(x_c,y)
        plt.plot(x_n,y)
    show_window(noticks=False,winmax=True)
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
    if os.path.exists(areaframe_ell_path) == True:
        img = np.load(areaframe_ell_path)
    running,count = True,0
    while running:
        p = get_points(img,test,'selectarea')[0]
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
    
        plt.imshow(img_new)
        show_window(noticks=True,winmax=True)
        count+=1
        usr = input('Would you like to continue selecting areas? (y/n)')
        if usr == 'n':
            running = False
        elif usr == 'y':
            img = img_new
    np.save(ell_filepath,ell_list)
    np.save(areaframe_ell_path,img_new)
    return

def edit_frame(img_new,ell,inf):
    c,w,h = inf[0],inf[1],inf[2]
    x = int(c[0]-w/2)
    y = int(c[1]-h/2)
    print(x,y)
    for i in range(int(w)):
        for j in range(int(h)):
            ispoint = ell.contains_point([x+i,y+j])
            if ispoint:
                img_new[y+j][x+i]=0
    return img_new

