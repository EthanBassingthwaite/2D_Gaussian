##import all the packages I might need
import numpy as np
import numpy.random as npr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
import astropy.units as u
from astropy.io import fits
from astropy import wcs
from astropy.io import ascii
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy.visualization import make_lupton_rgb
from astropy.modeling import models, fitting
#from reproject import reproject_interp
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
import sys
import os
import os.path
from os import path
import pandas as pd 
"""
    Johanna Mori
    8/16/17
    Fits a 2D Gaussian to the image whose size is a multiple of the user radius defined by 
    multiple_of_user_radius in the body of the program. Can fit with a peak location (mean) 
    that is free to move or fixed in the center, with x and y standard deviations that can 
    be forced to be equal or varied independently, or either a 2 component fit (Gaussian 
    plus a baseline) or one component (just the Gaussian).
    
    To use, make sure you have your file system set up right according to the file i/o section
    including the output folder for the images, change the settings and flags to whatever you
    would like them to be, enter the range of sources you want to fit, and run from the 
    command line.
    
    Summer 2020 Katie Devine modified to work on background subtracted images (post-photometry)
    removed 2-component fit options ofearlier code versions because no background
    
    August 2020 added uncertainty output of fitting 
    
    December 2022 Grace Wolf-Chase modified to only process fits files if both 8um & 24um images are present
    and to read in four-column input files (ID, GLON, GLAT, r)
    without categories.
    
    Conventions:
    Mips in red, Glimpse in green
    Longer dashes, longer wavelengths (8um gets dots, 24um gets dashes)
    fpa is flux per area
    #"""

#########################################################################################################
'''
    Settings
    #'''
#########################################################################################################

#Flags to only fit sources with specific properties
only_classic_ybs = False
only_not_classic_ybs = False
only_patchable_sources = False
only_point_plusses = False
only_baby_bubbles = False
sort_into_cut_folders = True
#2/15/23: GWC try sorting into cut folders - set equal to True
show_3D_residuals_in_plot = True 

#fit settings
fixed_mean = False #force the peak of the gaussian to be at the center of the user radius
equal_stddev = False #force a circular gaussian fit
multiple_of_user_radius = 2 #how large of an area to fit over
max_patch_size = 0 #How many pixels to patch before we declare it an oversaturated source

#Autofit parameters ## I basically semi-arbitrarily picked these so that the cuts looked okay. Feel free to pick better numbers
STDEV_TOLERANCE = 0.4
PEAK_DIFFERENCE_TOLERANCE = 0.3 #times the average std. dev?
RESID_TOLERANCE = 0.10


#File I/O setup
#12/22/22: Renaming input catalog to DR2-YBGLON30-40_wID.csv
#01/17/23: Created file DR2-YBGLON40-60_wID.csv to process YBs between GLON40-60
#01/19/23: Created file DR2-YB-SMOG_wID.csv to process SMOG data;
#SMOG data used cdelt rather than cd for coordinate increments, and this will need to be changed.
#01/19/23: Created DR2-YBs3504-3762_wID.csv to process YBs from GLON301.5-307.5
#GWC Current input catalogs use only 4 columns: ID, GLON, GLAT, r
#ID GLON GLAT r, no other columns
catalog_name = "DR2-YB-CygX_wID.csv" 
#catalog_name = "DR2-YBclass30-40_wID.csv" # 30-40 degree YB DR2 catalog, coords in galactic & degrees, Sarah's categories. Columns are:
#ID GLON	GLAT	r	Bubble	Classic/bubble rim	Classic/inside bubble	Filament/ISM morphology	Classic/on filament	Pillar/Point Source	Point source Plus	Faint fuzzy	Classic/faint fuzzy	Classic YB	Unsure	Notes

output_directory = './graphs2D/'
outfilename = output_directory + '2D_Gaussian_Fits.csv'

#########################################################################################################
'''
    Working Functions
    #'''
#########################################################################################################


def make_title(index):
    lon = str( round(data[index]['GLON'], 2))
    lat = str( round(data[index]['GLAT'], 2))
    YBID  = str( round(data[index]['ID'], 2))
    if data[index]['GLAT'] > 0:
        lat = '+' + lat
    return 'YB%s_G%s%s' % (YBID, lon, lat)

#12/22/22: Commenting out get_category, which we no longer use.
#def get_category(index):
#    ret_string = ''
#    for i in range(10):
#       if data[index][i+3] == "X":
#            ret_string = ret_string + str(i)
#    if ret_string == '':
#        ret_string = '-1' #marker for unclassified sources
#    return ret_string

def tied_stddev(model):
    x_stddev = model.y_stddev_0
    return x_stddev

def fit_2D_gaussian(index, image_to_fit, fit_range, usr_rr):
    #gaussian setup params
    amp_guess = np.max(image_to_fit)
    if equal_stddev:
        tied_parameters2 = {'x_stddev': tied_stddev}
        init1 = models.Gaussian2D(amplitude = amp_guess, x_stddev = usr_rr/2,\
                                      y_stddev = usr_rr/2, x_mean = 0, y_mean = 0, \
                                      tied = tied_parameters2)
    else:
        init1 = models.Gaussian2D(amplitude = amp_guess, x_stddev = usr_rr/2, y_stddev = usr_rr/2, x_mean = 0, y_mean = 0) #

    if fixed_mean: # forces center to be at the center of the user radius
        init1.x_mean.fixed = True
        init1.y_mean.fixed = True
    else: #make the center stay within the user radius square
        init1.x_mean.min = -usr_rr
        init1.x_mean.max =  usr_rr
        init1.y_mean.min = -usr_rr
        init1.y_mean.max =  usr_rr

    #force a non-negative fit
    init1.amplitude.min = 0.
    init1.x_stddev.min = 0.
    init1.y_stddev.min = 0.

    init1.x_stddev.max = image_to_fit.shape[0] #stops the gaussian from being too wide, and forces it to stay on the image
    init1.y_stddev.max = image_to_fit.shape[0]
    init = init1       
    fit = fitting.LevMarLSQFitter()
    x, y = np.mgrid[-fit_range:fit_range, -fit_range:fit_range] 
#    print(init1.param_names)
#    result: ('amplitude', 'x_mean', 'y_mean', 'x_stddev', 'y_stddev', 'theta')
    fitted=fit(init, x, y, image_to_fit)
    if fit.fit_info['param_cov'] is None:
        cov_diag=[0,0,0,0,0,0]
        print(cov_diag)
 #       cov_diag=np.NaN
 #       print(cov_diag)
    else:
        cov_diag = np.diag(fit.fit_info['param_cov'])
#    print(cov_diag)
#    print(fit.fit_info)
    return[x, y, fitted, cov_diag]

def dist_between(point0, point1):
    dist = np.sqrt((point1[0] - point0[0])**2 + (point1[1] - point0[1])**2)
    return dist

def similar_sddev(sd1, sd2):
    return abs((sd1-sd2)/((sd1+sd2)/2))

#########################################################################################################
#''' Hang onto your hats we're building a class'''
#########################################################################################################

class AstroImage:
#Attributes of the class:
#   rad: array of radii, filled in init
#   image_number: 30, 33, or 36; whichever mosaic you want
#   Added 39 on 6/25/18.
#   wv: Wavelength; either 8 or 24  -> no longer exists
#   image_data: array with the image in it
#   w: fits header
#   delta: a thing from the header which we need a lot but is in different places in the unsmoothed glimpse and mips
#   pixscale: size of the pixels in arcseconds
#   pixcoords: array of coordinates in pixels

    def __init__(self, image_number):
        image_name8='./fits_cutouts/8_umresid_YB_%s.fits' %(image_number)
        image_name24='./fits_cutouts/24_umresid_YB_%s.fits' %(image_number)
        
        self.hdu8_list=fits.open(image_name8)
        self.hdu24_list=fits.open(image_name24)
        self.image_data = {8 : self.hdu8_list[0].data, 24 : self.hdu24_list[0].data}
        self.w = wcs.WCS(self.hdu8_list[0].header)
 #       self.delta = self.w.wcs.cdelt[1]
 #       print(self.delta)
        self.delta = self.w.wcs.cd[1][1]
        self.pixscale=self.delta*3600 #about 1.2 arcsec per pixel
        self.pixcoords = np.empty((len(data), 2), np.float_)
        self.ybwcs = np.array([[data[0]['GLON'],data[0]['GLAT']]], np.float_)
        for n in range(0, len(data)):
            self.ybwcs = np.insert(self.ybwcs,n,[data[n]['GLON'],data[n]['GLAT']],axis=0)
            self.pixcoords = self.w.wcs_world2pix(self.ybwcs, 1)
        ########################################################################################

    def show_gaussian(self, index, f):  #this is the one that gets called from the main program

#12/22/22: Commenting out ifs that use get_category        
#        if only_classic_ybs:
#            if get_category(index) != '9':
#                return
    
#        if only_not_classic_ybs:
#            if get_category(index) == '9':
#                return
    
#        if only_point_plusses:
#            if get_category(index) != '6':
#                return
        
#        if only_baby_bubbles:
#            if get_category(index) != ''

        usr_rr = data[index]['r']/self.delta

        #fit_range = int(usr_rr * multiple_of_user_radius)
        fit_range = 50
        patch_flag = 0
        
#        image8 = self.image_data[8][yy-fit_range:yy+fit_range, xx-fit_range:xx+fit_range]   #doesn't check the 8um for saturation, because none of ours are saturated
        image8 = self.image_data[8]  #doesn't check the 8um for saturation, because none of ours are saturated

#        print(index, fit_range, usr_rr)
        x_axis, y_axis, fit8 , uncert8 = fit_2D_gaussian(index, image8, fit_range, usr_rr)            #but the image patcher works fine for both. Just set up an 8um flag and a 24um flag
        print(uncert8)
#        image24, bad_pixel_map, patch_flag = self.image_patcher(self.image_data[24][yy-fit_range:yy+fit_range, xx-fit_range:xx+fit_range], patch_flag)
        image24, bad_pixel_map, patch_flag = self.image_patcher(self.image_data[24], patch_flag)

        x_axis, y_axis, fit24, uncert24 = fit_2D_gaussian(index, image24, fit_range, usr_rr)
        print(uncert24)
        
        if only_patchable_sources:
            if patch_flag == -1:
                return
        
        resid8 = image8 - fit8(x_axis, y_axis)
        resid24 = image24 - fit24(x_axis, y_axis)
        center = [0,0]
        
        #if you want to add a cut like the only_patchable_sources cut, I would probably do it here, or as soon as
        #possible after you've computed all the things you need to check. That way you'll speed up the program as
        #much as possible by not doing any more math than necessary.
        
        r = np.zeros(image8.shape)
        g = np.zeros(image8.shape)
        b = np.zeros(image8.shape)
        

        mean8 = [fit8.y_mean.value + image8.shape[0]/2,  fit8.x_mean.value + image8.shape[0]/2]
        rmean8 = [fit8.y_mean.value, fit8.x_mean.value]
        width8 = fit8.x_stddev.value * 2
        height8 = fit8.y_stddev.value * 2
        angle8 = fit8.theta.value
        d8 = str(fit8.amplitude.value) +','+str(uncert8[0])+','+str(fit8.x_mean.value)+','+str(uncert8[1]) +','+ str(fit8.y_mean.value)+','+str(uncert8[2])
        
        mean24 = [fit24.y_mean.value + image24.shape[0]/2,  fit24.x_mean.value + image24.shape[0]/2]
        rmean24= [fit24.y_mean.value, fit24.x_mean.value]
        width24 = fit24.x_stddev.value * 2
        height24 = fit24.y_stddev.value * 2
        angle24 = fit24.theta.value
        d24 = str(fit24.amplitude.value)+','+str(uncert24[0])+','+str(fit24.x_mean.value)+','+str(uncert24[1])+','+ str(fit24.y_mean.value)+','+str(uncert24[2])

        d8 = d8 +', %s, %s, %s, %s, %s, %s, %s, ' %(dist_between(rmean8, center), fit8.x_stddev.value*self.delta*3600, uncert8[3]*self.delta*3600, fit8.y_stddev.value*self.delta*3600, uncert8[4]*self.delta*3600,angle8, uncert8[5])
        d24=d24 +', %s, %s, %s, %s, %s, %s, %s, ' %(dist_between(rmean24, center), fit24.x_stddev.value*self.delta*3600, uncert24[3]*self.delta*3600, fit24.y_stddev.value*self.delta*3600,  uncert24[4]*self.delta*3600, angle24, uncert24[5])
        
        ell8 = mpl.patches.Ellipse(xy=mean8, width=width8, height=height8, angle = angle8, linestyle = ':', fill = False)
        ell8w = mpl.patches.Ellipse(xy=mean8, width=width8, height=height8, angle = angle8, color = 'w', linestyle = ':', fill = False)
        ell24 = mpl.patches.Ellipse(xy=mean24, width=width24, height=height24, angle = angle24, fill = False, linestyle = '--')
        ell24w = mpl.patches.Ellipse(xy=mean24, width=width24, height=height24, angle = angle24, fill = False, color = 'w', linestyle = '--')
        
        usr_circle = Circle((image8.shape[0]/2, image8.shape[0]/2), data[index]['r'] / self.delta, \
                            fill=False)
        usr_circlew = Circle((image8.shape[0]/2, image8.shape[0]/2), data[index]['r'] / self.delta, \
                             color='w', fill=False)
        fig = plt.figure(figsize=(10,10))
        
        #creates the plots
        row = 3
        column = 3
        #rg image,
        ax = plt.subplot(row, column,1, title = 'Image data', projection = self.w)
        ax.add_patch(ell8)
        ax.add_patch(ell24)
        ax.add_patch(usr_circle)
        r = image24
        g = image8
        ax.imshow(make_lupton_rgb(r, g, b, stretch=200, Q=0), norm=LogNorm())
        
        #rg residuals
        ax2 = plt.subplot(row, column,2, title = 'residuals', projection = self.w)
        r = resid24
        g = resid8
        ax2.imshow(make_lupton_rgb(r, g, b, stretch=200, Q=0), norm=LogNorm())
        ax2.add_patch(ell8w)
        ax2.add_patch(ell24w)
        ax2.add_patch(usr_circlew)
        
        #plot of bad pixels and handy info
        #12/22/22: Removing get_category from ax3
        #ax3 = plt.subplot(row, column, 3, title = 'Patch flag:' + str(patch_flag)\
        #                  +'\n Category: '+ get_category(index), projection = self.w)    
        ax3 = plt.subplot(row, column, 3, title = 'Patch flag:' + str(patch_flag),\
                          projection = self.w)
        ax3.imshow(bad_pixel_map)
      
        #3d plot model 8um
        ax3 = plt.subplot(row, column, 4, projection = '3d', title = '8um model')
        surf = ax3.plot_surface(x_axis, y_axis, fit8(x_axis, y_axis))
        
        #3d plot data 8um
        ax4 = plt.subplot(row, column, 5, projection = '3d', title = '8um data')
        data3d = ax4.plot_surface(x_axis, y_axis, image8)

        #3d plot model 24um
        ax3 = plt.subplot(row, column, 7, projection = '3d', title = '24um model')
        surf = ax3.plot_surface(x_axis, y_axis, fit24(x_axis, y_axis))
        
        #3d plot data 24um
        ax4 = plt.subplot(row, column, 8, projection = '3d', title = '24um data')
        data3d = ax4.plot_surface(x_axis, y_axis, image24)

        if show_3D_residuals_in_plot:
            #3d plot residual 8um
            ax4 = plt.subplot(row, column, 6, projection = '3d', title = '8um residual')
            ax4.set_zlim3d(top=100)
            data3d = ax4.plot_surface(x_axis, y_axis, resid8)
    
            #3d plot residual 24um
            ax4 = plt.subplot(row, column, 9, projection = '3d', title = '24um residual') #fix z scale to 100
            ax4.set_zlim3d(top=100)
            data3d = ax4.plot_surface(x_axis, y_axis, resid24)
        else:
            #2d fit with 1 sigma ellipse 8um
            tester_mask = np.zeros(x_axis.shape)
            tester_image = fit8(x_axis, y_axis)
            
            randx = npr.random() * x_axis.shape[0]/2
            randy = npr.random() * x_axis.shape[0]/2
            randt = npr.random() * np.pi
            randsd= npr.random() * x_axis.shape[0]/6
            
            
            testgaussian = models.Gaussian2D(amplitude = 1, x_stddev = randsd,\
                              y_stddev = randsd * 3, x_mean = randx, y_mean = randy, theta = randt)
                            
            tge = mpl.patches.Ellipse(xy=[x_axis.shape[0]/2 + randy, x_axis.shape[0]/2 + randx], width=randsd*3, height=randsd, angle = randt * (-180.0)/np.pi, fill = False, color = 'w') #testing the ellipse plotting
            #print #yellowball WCS from data, find yellowball WCS here

            for i in range(x_axis.shape[0]):
                for j in range(x_axis.shape[1]):
                    #if tester_image[i, j] > tester_image.max() * 0.607:
                    #    tester_mask[i,j] = 0.5
                        if tester_image[i, j] == tester_image.max():
                            tester_mask[i,j] = 1
        
            ell8_2 = mpl.patches.Ellipse(xy=mean8, width=width8, height=height8, angle = angle8 * (-180.0/np.pi), linestyle = ':', fill = False) #testing the ellipse plotting
            ell24_2 = mpl.patches.Ellipse(xy=mean24, width=width24, height=height24, angle = angle24 * (-180.0/np.pi), fill = False, linestyle = '--') #testing the ellipse plotting
            
            #set max of tester image
            #Set all inside one sd to some color
            
            ax4 = plt.subplot(row, column, 3)
            ax4.imshow(testgaussian(x_axis, y_axis))#, cmap = 'gray')
            #ax4.add_patch(ell8_2)
            dline = np.zeros(x_axis.shape[0])
            for i in range(x_axis.shape[0]):
                dline[i] = i
            #ax4.plot(dline, dline)
            ax4.add_patch(tge)
            #ax4.add_patch(test_circx)
            #ax4.add_patch(test_circy)
            #ax4.add_patch(test_circb)
            
            #2d fit with 1 sigma ellipse 24um
            ax4 = plt.subplot(row, column, 6, projection = self.w)
            ax4.imshow(fit8(x_axis, y_axis)) #################FIX THIS
            ax4.add_patch(ell8_2)
            ax4.plot(dline, dline)
                
            #2d fit with 1 sigma ellipse 24um
            ax4 = plt.subplot(row, column, 9, projection = self.w)
            ax4.imshow(fit24(x_axis, y_axis)) #################FIX THIS
            ax4.add_patch(ell24_2)
            ax4.plot(dline, dline)

        #integrates the inputs and residuals, basically.
        totalresid24= 0
        totalresid8 = 0
        total8 = 0
        total24 =0
    
        #1/31/23: GWC - In the loop below, we will need to take the absolute value of the resid in each
        #pixel to avoid having very large + and - values add to deceptively small totalresid values.
        #2/15/23: GWC - absolute value doesn't lead to very useful results - it excludes too much,
        #so going back to original cuts. Would a better way to exclude wildly +/- residuals be to 
        #compare the fluctuations to the flux peak, or number of residual pixels that exceed a certain
        #fraction of the peak?
        for i in range(resid8.shape[0]):
            for j in range(resid8.shape[1]):
 #               totalresid8 += abs(resid8[i][j])
 #               totalresid24 += abs(resid24[i][j])
                totalresid8 += resid8[i][j]
                totalresid24 += resid24[i][j]
                total8 += image8[i][j] 
                total24+= image24[i][j]
 #               total8 += abs(image8[i][j])
 #               total24 += abs(image24[i][j])
        
        #calculuate the ratio of residual flux to original flux prior to unit conversion        
        residratio8 = abs(totalresid8/total8)
        residratio24 = abs(totalresid24/total24)
        
        #1/31/23: GWC - Note total8 and total24 are not used elsewhere, but these are the integrated
        #Gaussian fluxes. In principle, these could be compared to user photometry integrated fluxes.
        #Here I will print total, totalresid, and residratio numbers as a check on what the code is doing.
        #print('total8=', total8)
        #print('total24=', total24)
        #print('totalresid8=', totalresid8)
        #print('totalresid24=', totalresid24)
        #print('residratio8=', residratio8)
        #print('residratio24=', residratio24)
        ##          Unit Conversions Info         ##
        # 8 um:
        #GLIMPSE 8 um flux units are MJy/steradian
        #obtain the 8 um pixel scale from the image header 
        #this gives the degree/pixel conversion factor used for overdrawing circle
        #and flux conversions (this is x direction only but they are equal in this case)
        #8 um square degree to square pixel conversion-- x*y 
        sqdeg_tosqpix8=self.delta*self.delta
        #8 um steradian to pixel conversion (used to convert MJy/Pixel)
        #     will be used to convert to MJy
        str_to_pix8=sqdeg_tosqpix8*0.0003046118

        totalresid8=totalresid8*str_to_pix8*10**6
        totalresid24=totalresid24*str_to_pix8*10**6   
        #print('totalresid8=', totalresid8)
        #print('totalresid24=', totalresid24)
        
        #if you want to add a column to the spreadsheet to categorize things, I would write a function that returns
        #whatever you decide the categories are (I'd do numbers, easier to deal with in excel) and apends it to the
        #end of the all_data string, before the endline probably.
        
        cut_val, cut_string = self.auto_categorizer(fit8, fit24, resid8, resid24, residratio8, residratio24)
        
        source_metadata = '%s, %s, %s, %s' % (data[index]['ID'], data[index]['GLON'], data[index]['GLAT'], data[index]['r'])
        d8 =  d8 + ' %s, %s' % (totalresid8, residratio8)
        d24 = d24+ ' %s, %s' % (totalresid24, residratio24)
        
#        all_data = source_metadata+','+ d8 +','+ d24 +', %s, %s, '%(dist_between(rmean8, rmean24)*self.delta*3600, patch_flag) + get_category(index)
        all_data = source_metadata+','+ d8 +','+ d24 +', %s, %s, %s'%(dist_between(rmean8, rmean24)*self.delta*3600, patch_flag, cut_string)      
        all_data +=  ',%s \n' %(cut_val)      

        print(cut_val, cut_string)
        cuts_count[index] = cut_val
        
        if sort_into_cut_folders:
            cut_folder = 'Cut%s' % cut_val
            if not os.path.exists(output_directory + cut_folder): #Checks if the cut folder exists, makes a new one if not
                #print cut_folder
                os.makedirs(output_directory + cut_folder)
        else: cut_folder = ''

        f.write(all_data) # dump everything to the spreadsheet

        fig.suptitle(make_title(index), fontsize=20, y=0.99)
        #plt.tight_layout()
        fig.savefig(output_directory + cut_folder +'/'+ make_title(index) + '.png') #Dump the picture to an image file
        plt.close()

    def auto_categorizer(self, fit8, fit24, resid8, resid24, residratio8, residratio24):
    #classic YB cutoffs:
        good_attributes = '#' #this string lets you know which criteria were met 
        sum = 0
        average_stddev = (fit8.x_stddev + fit8.y_stddev + fit24.x_stddev + fit24.y_stddev)/ 4

#Is the residual/total of 8 um <10% ?
        if residratio8 <= 0.1:
            good_attributes += '1'
            sum += 1
        else: good_attributes += '0'

#Is the residual/total of 24 um <10% ?
        if residratio24 <= 0.1:
            good_attributes += '1'
            sum += 1
        else: good_attributes += '0'

#Is the distance between the center pixels less than the average standard devuation * peak difference tolerance defined above    
#Print distance between 8 & 24-micron centers
#        print('dist_between=', dist_between([fit8.x_mean.value, fit8.y_mean.value], [fit24.x_mean.value, fit24.y_mean.value]))    
        if dist_between([fit8.x_mean.value, fit8.y_mean.value], [fit24.x_mean.value, fit24.y_mean.value]) < average_stddev*PEAK_DIFFERENCE_TOLERANCE:
            good_attributes += '1'
            sum += 1
        else: good_attributes += '0'
        
        # if similar_sddev(fit8.x_stddev, fit8.y_stddev) < STDEV_TOLERANCE:
        #     good_attributes += '1'
        #     sum +=1
        # else: good_attributes += '0'

#Is the difference between the average 8um stdev and average 24 um stdev within tolerance defined above?   
#1/31/23: GWC - Print average standard deviations at 8 and 24 microns.     
        sdave8 = (fit8.x_stddev + fit8.y_stddev)/2
        sdave24= (fit24.x_stddev + fit24.y_stddev)/2
#        print('sdave8=', sdave8)
#        print('sdave24=', sdave24)
        if similar_sddev(sdave8, sdave24) < STDEV_TOLERANCE:
            good_attributes += '1'
            sum +=1
        else: good_attributes += '0'
        
        # if similar_sddev(fit24.x_stddev, fit24.y_stddev) < STDEV_TOLERANCE:
        #     good_attributes += '1'
        #     sum +=1
        # else: good_attributes += '0'
        
        # if np.nanmax(resid8) < RESID_TOLERANCE*fit8.amplitude:
        #     good_attributes += '1'
        #     sum += 1
        # else: good_attributes += '0'

        # if np.nanmax(resid24) < RESID_TOLERANCE*fit24.amplitude:
        #     good_attributes += '1'
        #     sum += 1
        # else: good_attributes += '0'

        return sum, good_attributes

    def image_patcher(self, image, patch_flag):
        num_bad_pixs = 0
        bad_pixels_x = list()
        bad_pixels_y = list()
        bad_pixel_map = np.zeros(image.shape)
        for i in range(image.shape[0]): #loop checks every pixel individually, not sure if there's a
            for j in range(image.shape[1]): #computationally better way to do it.
                if np.isnan(image[i][j]):
                    bad_pixels_x.append(i)
                    bad_pixels_y.append(j)
                    bad_pixel_map[i,j] = 1
                    num_bad_pixs += 1
        if num_bad_pixs == 0:
            return image, bad_pixel_map, patch_flag
        if num_bad_pixs > max_patch_size:
            patch_flag = -1
            return image, bad_pixel_map, patch_flag
        patch_value = np.nanmax(image)
        for i in range(len(bad_pixels_x)):
            for j in range(len(bad_pixels_y)):
                image[bad_pixels_x[i]][bad_pixels_y[i]] = patch_value
        #patches by filling nans in with the maximum array value. Change to whatever better patch you want
        patch_flag = num_bad_pixs
        return image, bad_pixel_map, patch_flag

#########################################################################################################
'''
    Actual Program Starts here
    #'''
#########################################################################################################

#reads files
data = ascii.read(catalog_name, delimiter = ',')

f = open(outfilename, 'w')
#if you add more columns make sure to add them here, otherwise topcat won't read the column headers right.
#Also, if you have a stray comma running around, it'll screw up the topcat headers. 
column_names = 'YB ID, l, b, User Radius, 8um Amplitude (MJy/Sr),D[8um Amplitude], \
8um x mean (pix), D[8um x mean ], 8um y mean (pix), D[8um y mean] ,8um dist from center (pix), 8um x sd dev (arcsec), D[8um x sd dev ], 8um y st dev (arcsec), D[8um y st dev], 8 um theta, D[8 um theta],\
8um residual (Jy), 8um residual/input flux (ratio), 24um amplitude (MJy/Sr), D[24um amplitude, 24um x mean (pix), D[24um x mean], 24um y mean (pix), D[24um y mean], 24um Dist from center (pix),\
24um x sd dev (arcsec), D[24um x sd dev], 24um y st dev (arcsec), D[24um y st dev ], 24um theta, D[24um theta], 24um residual (Jy), 24um residual/input flux ratio, \
dist between means (arcsec), Patch Flag, classification string, cut value\n'
f.write(column_names)

YB1=data['ID'][0]
YB2=YB1+len(data)
cuts_count = np.full(len(data), -1) 

#for i in range(1650,1655):
#careful with range here to select the YBs for which you have background-subtracted, post photometry images
#for i in range(1153,YB2):
for i in range(2547,YB2):    
    if path.exists('./fits_cutouts/8_umresid_YB_%s.fits' %(i)) == True and path.exists('./fits_cutouts/24_umresid_YB_%s.fits' %(i)) == True:
        twocolor_image30 = AstroImage(i)
        j=i-YB1
        twocolor_image30.show_gaussian(j, f)
        print(i,j)
        

plt.hist(cuts_count, range = [-0.5, 6.5], bins = 7, ec = 'black') #make sure you make it big enough to get all the cut categories, if any more get added
plt.savefig(output_directory  + 'histogram.png')
print('\a')



