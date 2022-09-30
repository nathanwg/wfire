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
##        plt.get_current_fig_manager().window.showMaximized()
        plt.get_current_fig_manager().window.state('zoomed')
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
##        usr = input('Save image, continue, or go back (s/c/b)')
##        if usr == 's':
##            continue
##        elif usr == 'c':
##            continue
##        elif usr == 'b':
##            return False
##        else:
##            continue

def get_heatmaps(testnums,data,findaverage,save,thresh):
    """.....
    """
    norm = 0
    norm = bool(norm)
    heat_maps = [];
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
##            plt.get_current_fig_manager().window.showMaximized()
##            clb.set_label('')
            plt.show()
    return True

##        for i in tests:
##        test = data[int(i)-1]
##        pathname = test.filename.replace('.tif','')
##        cwd = os.getcwd()
##        path = cwd+'\\heatmaps\\'+pathname+'_heatmap.tiff'
##        heatmap = readfile(path,False)[0]
##        plt.imshow(heatmap,cmap='nipy_spectral_r')
##        plt.show()

def get_points(img,data,test_num,points_type):
    """....
    """
    cwd = os.getcwd()
    test = data[int(test_num)-1]
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
    plt.get_current_fig_manager().window.state('zoomed')
    if points_type == 'grid':
        p = plt.ginput(n=-1,timeout=-1,show_clicks=True)
        p = refine_gridpoints(p)
    else:
        p = plt.ginput(num_points)
##    print(p[0][0],p[0][1])
    plt.close()

    return p,num_points,True

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

def load_heatmaps(tests,data):
    """...
    """
    heat_maps = [];
    for i in tests:
        test = data[int(i)-1]
        pathname = test.filename.replace('.tif','')
        cwd = os.getcwd()
        path = cwd+'\\heatmaps\\'+pathname+'_heatmap.npy'
        if os.path.exists(path) is False:
            print('---------------------------------------------------------\nNo heatmap file exists for this test number\n',i,'\n')
            usr = input('Ok (press return)')
            return 0
        heatmap = np.load(path)
        heat_maps.append(heatmap)
    return heat_maps


def save_points(test_num,data,p,points_type,num_points):
    """...
    """
    test = data[int(test_num)-1]
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
##    for i in range(0,num_cols):
##        x = round(x_L+i*dx)
##        y = round(y_c)
##        if y >= 256:
##            y = 255
##        if x>=256:
##            x = 255
##        h[y,x] = 4500
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
##    distance[1] = int(input('What distance to the left would you like to set? (cm): '))
##    distance[2] = int(input('What distance to the right would you like to set? (cm): '))
    return distance

def plotprofiles_h(sets,data,distance,isnorm,ylim):
    labels = []
    for i in range(0,len(sets),2):
        start = int(sets[i])
        stop = int(sets[i+1])
        tests = np.linspace(start,stop,stop-start+1)
        heat_maps = load_heatmaps(tests,data)
        temperature = int(data[int(tests[0])-1].set_type[3])
        temp_label = 'avg T = '+str(temperature)+' C'
        labels.append(temp_label)
        coordinates = get_line_coordinates(tests,data,distance)
        
        if heat_maps is 0 or coordinates is 0:
            return
        L = coordinates[0,3]
        line_avg = 0
        line_avg_norm = 0
        labels = ['avg T = 460','avg T = 520','avg T = 610','avg T = 670','avg T = 880']
        num_cols = heat_maps[0].shape[1]
        x_plot = np.linspace(0,9,num_cols)
        num = len(heat_maps)
        for ii in range(0,num):
            num_cols = heat_maps[0].shape[1]
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
                line[0,j] = heat_maps[ii][y,j]

            test = data[int(tests[ii])-1]
            eof = test.eof
            line_norm = line/(eof-test.ignition_time[1])
            line_avg += line
            line_avg_norm += line_norm
##        print('Num: ',num)
##        input('Continue....')
        line_avg /= num
##        print('Line average',line_avg)
        line_avg_norm /= num
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
        heat_maps = load_heatmaps(tests,data)
        temperature = int(data[int(tests[0])-1].set_type[3])
        temp_label = 'avg T = '+str(temperature)+' C'
        labels.append(temp_label)
##        print(temperature)
##        input('Correct temp?')
        if heat_maps is 0:
            return
        num_rows = heat_maps[0].shape[0]
        line_avg = 0
        line_avg_norm = 0
##        labels = ['avg T = 460','avg T = 520','avg T = 610','avg T = 670','avg T = 880']
        y_plot = np.linspace(0,9,num_rows)
        num = len(heat_maps)
        for ii in range(0,num):
            test = data[int(tests[ii])-1]
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
                line[0,-1*j+num_rows-1] = heat_maps[ii][j,x]
            eof = test.eof
            line_norm = line/(eof-test.ignition_time[1])
            line_avg += line
            line_avg_norm += line_norm
##        print('Num: ',num)
##        input('Continue....')
        line_avg /= num
##        print('Line average',line_avg)
        line_avg_norm /= num
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

def get_flametimeline(sets,data):
    for i in range(0,len(sets),2):
        start = int(sets[i])
        stop = int(sets[i+1])
        tests = np.linspace(start,stop,stop-start+1)
        coordinates = get_line_coordinates(tests,data,[2,2,2])
        
        if coordinates is 0:
            return
        for j in range(0,len(tests)):
            test = data[int(tests[j])-1]
            cwd = os.getcwd()
            pfilepath = cwd+'\\points_timeline\\'+test.filename.replace('.tif','')+'_points_timeline.txt'
            if os.path.exists(pfilepath) is False:
                print('---------------------------------------------------------\nNo points file exists for this test number\n',tests[j],'\n')
                usr = input('Ok (press return)')
                return
            p = np.loadtxt(pfilepath,unpack=True)
##            x_val = int(p[0])
##            y_val = int(coordinates[j,0])
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
            for jjj in range(num_points):
                if jjj <= num_points/2:
                    color = 'k'
                    level = 1
                else:
                    color = 'b'
                    level = -1
##                for k in range(ignition_frame,eof):
                for k in range(frame_span):
                    frame = frames[k+ignition_frame].astype(float)
                    if frame[y_val[jjj],x_val[jjj]] > 35:
                        timeline[jjj,k] = level
        ##            x_time = np.linspace(ignition_frame,eof,frame_span)
                plt.plot(x_time,timeline[jjj,:].T,color)
##            plt.show()
            if os.path.exists(save_filepath):
                print('---------------------------------------------------------\nLooks like there\'s already a file with this name.\nDelete existing file if you are wanting to overwrite\n')
                print(pfilepath)
                usr = input('Ok (press return)')
                return
            plt.savefig(save_filepath)
            plt.close()

def saveframes(sets,data):
    for i in range(0,len(sets),2):
        start = int(sets[i])
        stop = int(sets[i+1])
        tests = np.linspace(start,stop,stop-start+1)

        for j in range(0,len(tests)):
            test = data[int(tests[j])-1]
            file = os.getcwd().replace('wfire','') + test.filename
            img,filename = readfile(file,True)
            frames,num_frames,num_rows,num_cols = get_image_properties(img)
            sfilepath = os.getcwd()+'\\frames\\test_frames.npy'
            np.save(sfilepath,frames)

            
def calc_avgint(sets,data,threshold):
    """ Calculates the average light intensity (for pixels above threshold)
    """
    for i in range(0,len(sets),2):
        start = int(sets[i])
        stop = int(sets[i+1])
        tests = np.linspace(start,stop,stop-start+1)

        for j in range(0,len(tests)):
            check = True
            test = data[int(tests[j])-1]
            file = os.getcwd().replace('wfire','') + test.filename
            img,filename = readfile(file,True)
            frames,num_frames,num_rows,num_cols = get_image_properties(img)
            avgint = np.zeros((num_frames,1))

            cwd = os.getcwd()
            pfilepath = cwd+'\\points_grid\\'+test.filename.replace('.tif','')+'_points_grid.txt'
            if os.path.exists(pfilepath) is False:
                print('---------------------------------------------------------\nNo points file exists for this test number\n',tests[j],'\n')
                usr = input('Ok (press return)')
                return
            p = np.loadtxt(pfilepath)
            x_left,x_right,y_bot,y_top = int(p[0]),int(p[1]+1),int(p[2]+1),int(p[3])

            #####---------------------
##            fig,ax=plt.subplots()
##            ims = []


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
                #####----------------
##                im=ax.imshow(y_mod)
##                ims.append([im])
##            ani = animation.ArtistAnimation(fig,ims,interval=50,blit=True,repeat_delay=1000)
##            plt.show()
                if check == False:
##                    plt.imshow(ref_frame,cmap='nipy_spectral_r')
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
##            plt.get_current_fig_manager().window.state('zoomed')
##            total_cat = cat[0]+cat[1]+cat[2]
##            for k in range(0,len(cat)):
##                cat[k] = round((cat[k]/total_cat)*100)
##                
##            mylabels = ['0-85: '+str(cat[0])+'%','85-170: '+str(cat[1])+'%','170-255: '+str(cat[2])+'%']
##            plt.pie(cat,labels=mylabels)
####            plt.show()
####            plt.close()
            save_filepath = cwd+'\\plots_avgint\\'+test.filename.replace('.tif','')+'_avgint.png'
            if os.path.exists(save_filepath):
                print('---------------------------------------------------------\nLooks like there\'s already a file with this name.\nDelete existing file if you are wanting to overwrite\n')
                print(pfilepath)
                usr = input('Ok (press return)')
                return
            plt.savefig(save_filepath)
            plt.close()
##            print(total,numpixels_frame)
##            input()
##            plt.show()

def creategrids(sets,data):
    sector_width = 30
    for i in range(0,len(sets),2):
        start = int(sets[i])
        stop = int(sets[i+1])
        tests = np.linspace(start,stop,stop-start+1)

        heat_maps = load_heatmaps(tests,data)

        for j in range(0,len(heat_maps)):
            test = data[int(tests[j])-1]
            cwd = os.getcwd()
            pfilepath = cwd+'\\points_grid\\'+test.filename.replace('.tif','')+'_points_grid.txt'
            if os.path.exists(pfilepath) is False:
                print('---------------------------------------------------------\nNo points file exists for this test number\n',tests[j],'\n')
                usr = input('Ok (press return)')
                return
            p = np.loadtxt(pfilepath)
            x_left,x_right,y_bot,y_top = p[0],p[1],p[2],p[3]
            img = heat_maps[j]
            num_rows = img.shape[0]
            calib = np.zeros((num_rows,1))
            calib[0,0] = 4500
            img = np.concatenate((img,calib),axis=1)

##            print(x_left,x_right,'\n',y_top,y_bot)
            num_x = int((x_right-x_left)/sector_width)+2
            num_y = int((y_bot-y_top)/sector_width)+2
            
            x_ticks,y_ticks = [],[]
            for k in range(0,num_x):
                x = k*sector_width+x_left
                x_ticks.append(x)
                if x >= x_right:
                    x_ticks[k] = x_right
                    break
##            print(x_ticks)
            for k in range(0,num_y):
                y = k*sector_width+y_top 
                y_ticks.append(y)
                if y >= y_bot:
                    y_ticks[k] = y_bot
                    break
##            print(y_ticks)
##            input()
            x_plot = np.linspace(x_left,x_right,100)
            plt.imshow(img,cmap='nipy_spectral_r')
            plt.get_current_fig_manager().window.state('zoomed')
            for k in y_ticks:
                y_plot = np.linspace(k,k,100)
                plt.plot(x_plot,y_plot,'k')
            y_plot = np.linspace(y_top,y_bot,100)
            for k in x_ticks:
                x_plot = np.linspace(k,k,100)
                plt.plot(x_plot,y_plot,'k')
            plt.show()
            
    return
                
                
