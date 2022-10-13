from cProfile import label
from cgi import test
from heapq import nsmallest
from json import load
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
        ig_test = Test(filename,(month[i],day[i],year[i]),testnum[i],(set_[i],orientation[i],height[i],temp[i]),fmc[i],(time[i],int(frame[i])),spatial[i],eof[i])
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
        plt.get_current_fig_manager().window.showMaximized()
        plt.show()

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

def show_ignition(testnums,data):
    """.....
    """
    for i in testnums:
        test = data[int(i)-1]
        file = os.getcwd().replace('wfire','') + test.filename
        img = readfile(file,True)[0]
        frames = img[1]
        show_frames(frames,test.ignition_time[1],test.eof)

def get_heatmaps(testnums,data,findaverage,save,thresh):
    """.....
    """
    norm = 0
    norm = bool(norm)
    heat_maps = []
    counter = 0
    for i in testnums:
        test = data[int(i)-1]
        print(test.testnumber)
        file = os.getcwd().replace('wfire','') + test.filename
        img,filename = readfile(file,True)
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
        heat_maps.append(heatmap)
        if save is True:
            name = test.filename.replace('.tif','')
            cwd = os.getcwd()
            cwd = cwd+'\\heatmaps\\'+name+'_heatmap.npy'
            if os.path.exists(cwd):
                print('---------------------------------------------------------\nLooks like there\'s already a file with this name. Delete existing file if you are wanting to overwrite\n')
                usr = input('Ok (press return)')
                continue
            np.save(cwd,heatmap)
        counter+=1
    
    if findaverage is True:
        avg_h = 0
        for i in range(0,len(heat_maps)):
            avg_h+=heat_maps[i]
        avg_h/=len(heat_maps)
        
    return heat_maps

def displaymaps(heat_maps,start,stop):
    """...
    """
    start,stop = int(start),int(stop)
    for i in range(start,stop+1):
        if heat_maps is None:
            input('Hello there')
            return False
        else:
            heatmap = heat_maps[i-start]
            usr = input('Continue (\'b\' to go back)')
            if usr == 'b':
                return False
            num_rows = heatmap.shape[0]
            calib = np.zeros((num_rows,1))
            calib[0,0] = 4500
            imgs = np.concatenate((heatmap,calib),axis=1)
            plt.imshow(imgs,cmap='nipy_spectral_r')
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.get_current_fig_manager().window.showMaximized()
            plt.show()
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
    if os.path.exists(pfilepath):
        print('---------------------------------------------------------\nLooks like there\'s already a file with this name.\nDelete existing file if you are wanting to overwrite\n')
        print(pfilepath)
        usr = input('Ok (press return)')
        return 0,0,False
    num_rows = img.shape[0]
    calib = np.zeros((num_rows,1))
    calib[0,0] = 4500
    img = np.concatenate((img,calib),axis=1)
    plt.imshow(img,cmap='nipy_spectral_r')
    plt.get_current_fig_manager().window.showMaximized()
    if points_type == 'grid':
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
    n = input('How many sets do you want to specify? - ')
    sets = []
    for i in range(0,int(n)):
        m1 = input('Enter start number of set: ')
        m2 = input('Enter end number of set: ')
        sets.append(int(m1))
        sets.append(int(m2))
    return sets

def load_heatmap(test):
    """...
    """
    pathname = test.filename.replace('.tif','')
    cwd = os.getcwd()
    path = cwd+'\\heatmaps\\'+pathname+'_heatmap.npy'
    if os.path.exists(path) is False:
        print('---------------------------------------------------------\nNo heatmap file exists for this test number\n',test.testnumber,'\n')
        usr = input('Ok (press return)')
        return 0
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
    if os.path.exists(pfilepath):
        print('---------------------------------------------------------\nLooks like there\'s already a file with this name.\nDelete existing file if you are wanting to overwrite\n')
        print(pfilepath)
        usr = input('Ok (press return)')
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

def get_line_coordinates(tests,data,d):
    coordinates = np.zeros((len(tests),4))
    k = 0
    conv_fact = data[int(tests[0])-1].spatial_calibration
    for ii in range(0,len(d)):
        d[ii] /= 100
        d[ii] /= conv_fact
        d[ii] = round(d[ii])
    for i in tests:
        test = data[int(i)-1]
        pathname = test.filename.replace('.tif','')
        cwd = os.getcwd()
        path = cwd+'\\points\\'+pathname+'_points.txt'
        if os.path.exists(path) is False:
            print('---------------------------------------------------------\nNo points file exists for this test number\n',i,'\n')
            usr = input('Ok (press return)')
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
##        coordinates[k,0],coordinates[k,1],coordinates[k,2],coordinates[k,3] = y_c,x_L,dx,L
        coordinates[k,0],coordinates[k,1],coordinates[k,2],coordinates[k,3] = y_c,x_L,1,L
        k += 1
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
    plt.show()

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
        coordinates = get_line_coordinates(tests,data,distance)
        
        if coordinates is 0:
            return
        L = coordinates[0,3]
        line_avg = 0
        line_avg_norm = 0
        labels = ['avg T = 460','avg T = 520','avg T = 610','avg T = 670','avg T = 880']
        num = len(tests)
        for ii in range(0,num):
            test = data[int(tests[ii])-1]
            heatmap = load_heatmap(test)
            if heatmap is 0:
                return
            num_cols = heatmap.shape[1]
            line = np.zeros((1,num_cols))
            y_c,x_L,dx = coordinates[ii,0],coordinates[ii,1],coordinates[ii,2]
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
    plt.show()
    plt.close()

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
            if heatmap is 0:
                return
            num_rows = heatmap.shape[0]
            pathname = test.filename.replace('.tif','')
            cwd = os.getcwd()
            path = cwd+'\\points_vert\\'+pathname+'_points_vert.txt'
            if os.path.exists(path) is False:
                print('---------------------------------------------------------\nNo points_vert file exists for this test number\n',tests[ii],'\n')
                usr = input('Ok (press return)')
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
    plt.show()
    plt.close()
        
def change_ylim(ylim):
    ylim = input('Set ylim for plotting: ')
    return ylim

def get_flametimeline(test):        
    cwd = os.getcwd()
    pfilepath = cwd+'\\points_timeline\\'+test.filename.replace('.tif','')+'_points_timeline.txt'
    if os.path.exists(pfilepath) is False:
        print('---------------------------------------------------------\nNo points file exists for this test number\n',test.testnumber,'\n')
        usr = input('Ok (press return)')
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
    if os.path.exists(save_filepath):
        print('---------------------------------------------------------\nLooks like there\'s already a file with this name.\nDelete existing file if you are wanting to overwrite\n')
        print(save_filepath)
        usr = input('Ok (press return)')
        return
    plt.savefig(save_filepath)
    plt.close()
    return

def creategrids(test):
    sector_width = 30
    heatmap = load_heatmap(test)
    cwd = os.getcwd()
    pfilepath = cwd+'\\points_grid\\'+test.filename.replace('.tif','')+'_points_grid.txt'
    if os.path.exists(pfilepath) is False:
        print('---------------------------------------------------------\nNo points file exists for this test number\n',test.testnumber,'\n')
        usr = input('Ok (press return)')
        return
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
    plt.show()
            
    return
                
                
def get_median(test):
    cwd = os.getcwd()
    filepath = cwd+'\\data_timeline\\'+test.filename.replace('.tif','')+'_timeline.txt'
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

def plotmedians(sets,data,medians_sets):
    showunc = True
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
    plt.show()
    return

def get_max_flame_area(test):
    x_left,x_right,y_bot,y_top = load_gridpoints(test)
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
    for k in range(num_frames):
        ref_frame = frames[k]
        ref_frame = ref_frame.astype(float) # Change to float to handle subtraction correctly
        ref_frame[y_top:y_bot,x_left:x_right] = 0
        if ref_frame.max() < threshold:
            continue
        x_bool = ((ref_frame-threshold)>=0)
        numpixels_frame = x_bool.sum()
        pixel_area = pixel_length**2
        flame_area = pixel_area*numpixels_frame*(1000**2)
        if flame_area > max_flame_area:
            max_flame_area = flame_area
            m_frame = ref_frame
            npf = numpixels_frame
            frame_num = k

    # for k in range(num_cols):
    #     for m in range(num_rows):
    #         if m_frame[m,k] > threshold:
    #             m_frame[m,k] = 255
    print('Numpixels_frame: ',npf,'\n','Pixel_area: ',pixel_area)
    print('Threshold: ',threshold)
    print('Total area: ',round(max_flame_area),' mm^2')
    # input()
    mfilepath = os.getcwd()+'\\flamearea\\'+test.filename.replace('.tif','')+'_flamearea_frame.npy'
    afilepath = os.getcwd()+'\\flamearea\\'+test.filename.replace('.tif','')+'_flamearea.npy'
    np.save(mfilepath,m_frame)
    np.save(afilepath,[max_flame_area,frame_num])
    # plt.imshow(m_frame)
    # plt.show()
                

def get_ima(test):
    mfilepath = os.getcwd()+'\\flamearea\\'+test.filename.replace('.tif','')+'_flamearea_frame.npy'
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
    if os.path.exists(pfilepath) is False:
        print('---------------------------------------------------------\nNo points file exists for this test number\n',test.testnumber,'\n')
        usr = input('Ok (press return)')
        return
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

def load_area(test):
    afilepath = os.getcwd()+'\\flamearea\\'+test.filename.replace('.tif','')+'_flamearea.npy'
    vals = np.load(afilepath)
    max_flamearea,frame_num = vals[0]/(10**2),vals[1]
    return max_flamearea,frame_num

def plot_max_flame_area(sets,data,max_flamearea_sets):  
    showunc = False
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
    plt.show()
    return

def plot_ima(sets,data,ima_sets):
    showunc = True
    labels,temperatures,linestyle = get_plotinfo(sets,data) 
    ima_averages = []
    for i in range(len(ima_sets)):        
        ima_averages.append(np.mean(ima_sets[i]))
        unc,cap = calc_uncertainty(ima_sets[i],10),4
        if showunc == False:
            unc,cap = 0,0
        plt.errorbar(temperatures[i],ima_averages[i]/255,fmt=linestyle[i],yerr=unc/255,capsize=cap)
    plt.xlabel('Average exhaust gas temperature $^{\circ}C$')
    plt.ylabel('Average intensity of max flame area')
    plt.title('Average normalized light intensity of maximum flame area')
    plt.show()
    return

def get_plotinfo(sets,data):
    labels,temperatures,linestyle = [],[],[]
    labeldried,labellive=False,False
    for i in range(0,len(sets),2):
        start = int(sets[i])
        stop = int(sets[i+1])
        tests = np.linspace(start,stop,stop-start+1)
        test = data[int(tests[0])-1]
        temperature = int(test.set_type[3])
        temp_label = 'avg T = '+str(temperature)+' C'
        temperatures.append(temperature)
        if test.fmc == 0:
            linestyle.append('ko')
            if labeldried == True:
                labels.append(None)
            elif labeldried == False:
                labels.append('Oven-dried fuel')
                labeldried = True
        else:
            linestyle.append('g^')
            if labellive == True:
                labels.append(None)
            elif labellive == False:
                labels.append('Live fuel')
                labellive = True
    return labels,temperatures,linestyle