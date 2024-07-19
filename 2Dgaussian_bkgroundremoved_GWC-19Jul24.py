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
import cv2
from scipy import interpolate
from astropy.nddata import Cutout2D
import copy
import ast
from matplotlib.colors import SymLogNorm

#This code is necesary for the google drive api
from pydrive2.auth import GoogleAuth
from oauth2client.service_account import ServiceAccountCredentials
from pydrive2.drive import GoogleDrive
from io import BytesIO

#Get the authentication and access to a proxy account
#basically I made a 'fake' google account for the computer to access
gauth = GoogleAuth()
SCOPES = ['https://www.googleapis.com/auth/drive']

gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name('client_secrets.json', SCOPES)
drive = GoogleDrive(gauth)




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

#Global variables so that the code only needs to load the csv file once
csvname = 'YBphotometry_results_WolfChase1-ALL_YBs.csv'
userdata = ascii.read(csvname, delimiter=',')

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
#catalog_name = "DR2-YB-SMOG_wID.csv" #"DR2-YB-CygX_wID.csv" 
#catalog_name = "DR2-YBclass30-40_wID.csv" # 30-40 degree YB DR2 catalog, coords in galactic & degrees, Sarah's categories. Columns are:
#ID GLON	GLAT	r	Bubble	Classic/bubble rim	Classic/inside bubble	Filament/ISM morphology	Classic/on filament	Pillar/Point Source	Point source Plus	Faint fuzzy	Classic/faint fuzzy	Classic YB	Unsure	Notes

output_directory = '2DGaussianOutputs/'
outfilename = output_directory + '2D_Gaussian_Fits.csv'

#########################################################################################################
'''
    Working Functions
'''
#########################################################################################################


def make_title(index):
    '''lon = str( round(data[index]['GLON'], 2))
    lat = str( round(data[index]['GLAT'], 2))
    YBID  = str( round(data[index]['ID'], 2))
    if data[index]['GLAT'] > 0:
        lat = '+' + lat
    return 'YB%s_G%s%s' % (YBID, lon, lat)'''
    # Minus 1 because index is the YBID, which is off by 1 from the actual index. 
    lon = str( round(data[index-1]['l'], 2))
    lat = str( round(data[index-1]['b'], 2))
    YBID  = str( round(data[index-1]['YB'], 2))
    if data[index-1]['b'] > 0:
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
   
    #fit = fitting.LevMarLSQFitter()
    fit = fitting.TRFLSQFitter()
    
    x, y = np.mgrid[-fit_range:fit_range, -fit_range:fit_range] 
#    print(init1.param_names)
#    result: ('amplitude', 'x_mean', 'y_mean', 'x_stddev', 'y_stddev', 'theta')
    #print(x, y)
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

    def __init__(self, image_number, Mosaics):
        
        '''image_name8='./fits_cutouts/8_umresid_YB_%s.fits' %(image_number)
        image_name24='./fits_cutouts/24_umresid_YB_%s.fits' %(image_number)
        
        self.hdu8_list=fits.open(image_name8)
        self.hdu24_list=fits.open(image_name24)
        print("This is what you are looking for")
        #print(self.hdu8_list[0].data[50])
        self.image_data = {8 : self.hdu8_list[0].data, 24 : self.hdu24_list[0].data}
        self.w = wcs.WCS(self.hdu8_list[0].header)
        print(self.hdu8_list[0].header)
        self.delta = self.w.wcs.cdelt[1]
        print(self.delta)
        self.delta = self.w.wcs.cd[1][1]'''
        
        
        
        self.residuals = get_residual(image_number, Mosaics)
        if self.residuals.skip8 or self.residuals.skip24:
            self.skip = True
        else: self.skip = False
        self.image_data ={ 8 : self.residuals.resid8, 24 : self.residuals.resid24}
        self.delta = 0.000333333
        self.w = wcs.WCS(self.residuals.w)
        
        
        
        
        self.pixscale=self.delta*3600 #about 1.2 arcsec per pixel
        self.pixcoords = np.empty((len(data), 2), np.float_)
        #self.ybwcs = np.array([[data[0]['GLON'],data[0]['GLAT']]], np.float_)
        self.ybwcs = np.array([[data[0]['l'],data[0]['b']]], np.float_)
        for n in range(0, len(data)):
            #self.ybwcs = np.insert(self.ybwcs,n,[data[n]['GLON'],data[n]['GLAT']],axis=0)
            self.ybwcs = np.insert(self.ybwcs,n,[data[n]['l'],data[n]['b']],axis=0)
            self.pixcoords = self.w.wcs_world2pix(self.ybwcs, 1)


            

    def show_gaussian(self, index):
        #If photometry was not conducted for either 8 or 24, then do nothing
        if self.residuals.skip8 and self.residuals.skip24:
            return 0


        # Minus 1 to match YBID to index position
        usr_rr = data[index-1]['r']/self.delta

        #fit_range = int(usr_rr * multiple_of_user_radius)
        fit_range = 50
        patch_flag = 0
        center = [0,0]
        #{
        image8 = self.image_data[8]  #doesn't check the 8um for saturation, because none of ours are saturated
        x_axis, y_axis, fit8 , uncert8 = fit_2D_gaussian(index, image8, fit_range, usr_rr)            #but the image patcher works fine for both. Just set up an 8um flag and a 24um flag
        print(uncert8)
        resid8 = image8 - fit8(x_axis, y_axis)

        mean8 = [fit8.y_mean.value + image8.shape[0]/2,  fit8.x_mean.value + image8.shape[0]/2]
        rmean8 = [fit8.y_mean.value, fit8.x_mean.value]
        width8 = fit8.x_stddev.value * 2
        height8 = fit8.y_stddev.value * 2
        angle8 = fit8.theta.value
        d8 = str(fit8.amplitude.value) +','+str(uncert8[0])+','+str(fit8.x_mean.value)+','+str(uncert8[1]) +','+ str(fit8.y_mean.value)+','+str(uncert8[2])
        d8 = d8 +', %s, %s, %s, %s, %s, %s, %s, ' %(dist_between(rmean8, center), fit8.x_stddev.value*self.delta*3600, uncert8[3]*self.delta*3600, fit8.y_stddev.value*self.delta*3600, uncert8[4]*self.delta*3600,angle8, uncert8[5])
        ell8 = mpl.patches.Ellipse(xy=mean8, width=width8, height=height8, angle = angle8, linestyle = ':', fill = False)
        ell8w = mpl.patches.Ellipse(xy=mean8, width=width8, height=height8, angle = angle8, color = 'w', linestyle = ':', fill = False)
        usr_circle = Circle((image8.shape[0]/2, image8.shape[0]/2), data[index-1]['r'] / self.delta, \
                            fill=False)
        usr_circlew = Circle((image8.shape[0]/2, image8.shape[0]/2), data[index-1]['r'] / self.delta, \
                             color='w', fill=False)
            
        totalresid8 = resid8.sum()
        total8 = image8.sum()
        
        #calculuate the ratio of residual flux to original flux prior to unit conversion        
        residratio8 = abs(totalresid8/total8)
        
        
        #}
        
        ########################################################################################################################
        
        #{
        image24 = self.image_data[24]
        x_axis, y_axis, fit24, uncert24 = fit_2D_gaussian(index, image24, fit_range, usr_rr)
        print(uncert24)
        resid24 = image24 - fit24(x_axis, y_axis)
        mean24 = [fit24.y_mean.value + image24.shape[0]/2,  fit24.x_mean.value + image24.shape[0]/2]
        rmean24= [fit24.y_mean.value, fit24.x_mean.value]
        width24 = fit24.x_stddev.value * 2
        height24 = fit24.y_stddev.value * 2
        angle24 = fit24.theta.value
        d24 = str(fit24.amplitude.value)+','+str(uncert24[0])+','+str(fit24.x_mean.value)+','+str(uncert24[1])+','+ str(fit24.y_mean.value)+','+str(uncert24[2])
        d24=d24 +', %s, %s, %s, %s, %s, %s, %s, ' %(dist_between(rmean24, center), fit24.x_stddev.value*self.delta*3600, uncert24[3]*self.delta*3600, fit24.y_stddev.value*self.delta*3600,  uncert24[4]*self.delta*3600, angle24, uncert24[5])
        ell24 = mpl.patches.Ellipse(xy=mean24, width=width24, height=height24, angle = angle24, fill = False, linestyle = '--')
        ell24w = mpl.patches.Ellipse(xy=mean24, width=width24, height=height24, angle = angle24, fill = False, color = 'w', linestyle = '--')
        totalresid24 = resid24.sum()
        total24 = image24.sum()
        
        #calculuate the ratio of residual flux to original flux prior to unit conversion        
        residratio24 = abs(totalresid24/total24)
        
        
        #}
        
        
        
        
        
        
        #if you want to add a cut like the only_patchable_sources cut, I would probably do it here, or as soon as
        #possible after you've computed all the things you need to check. That way you'll speed up the program as
        #much as possible by not doing any more math than necessary.
        
        
        #This is where the code for making the csv file happens
        #{
        #integrates the inputs and residuals, basically.
        #totalresid24 = 0
        #totalresid8 = 0
        #total8 = 0
        #total24 = 0
    
        #1/31/23: GWC - In the loop below, we will need to take the absolute value of the resid in each
        #pixel to avoid having very large + and - values add to deceptively small totalresid values.
        #2/15/23: GWC - absolute value doesn't lead to very useful results - it excludes too much,
        #so going back to original cuts. Would a better way to exclude wildly +/- residuals be to 
        #compare the fluctuations to the flux peak, or number of residual pixels that exceed a certain
        #fraction of the peak?
        
        #for i in range(resid8.shape[0]):
        #    for j in range(resid8.shape[1]):
 #      #         totalresid8 += abs(resid8[i][j])
 #      #         totalresid24 += abs(resid24[i][j])
        #        totalresid8 += resid8[i][j]
        #        totalresid24 += resid24[i][j]
        #        total8 += image8[i][j] 
        #        total24+= image24[i][j]
 #               total8 += abs(image8[i][j])
 #               total24 += abs(image24[i][j])
        
        
        
        #calculuate the ratio of residual flux to original flux prior to unit conversion        
        #residratio8 = abs(totalresid8/total8)
        #residratio24 = abs(totalresid24/total24)
        #1/31/23: GWC - Note total8 and total24 are not used elsewhere, but these are the integrated
        #Gaussian fluxes. In principle, these could be compared to user photometry integrated fluxes.

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

        
        #if you want to add a column to the spreadsheet to categorize things, I would write a function that returns
        #whatever you decide the categories are (I'd do numbers, easier to deal with in excel) and apends it to the
        #end of the all_data string, before the endline probably.
        
        cut_val, cut_string = self.auto_categorizer(fit8, fit24, resid8, resid24, residratio8, residratio24)
        print(cut_string, "cutstring")
        #source_metadata = '%s, %s, %s, %s' % (data[index]['ID'], data[index]['GLON'], data[index]['GLAT'], data[index]['r'])
        
        # Minus 1 to match YBID to index position
        source_metadata = '%s,%s,%s' % (data[index-1]['l'], data[index-1]['b'], data[index-1]['r'])
        d8 =  d8 + '%s,%s' % (totalresid8, residratio8)
        d24 = d24+ '%s,%s' % (totalresid24, residratio24)
        
#        all_data = source_metadata+','+ d8 +','+ d24 +', %s, %s, '%(dist_between(rmean8, rmean24)*self.delta*3600, patch_flag) + get_category(index)
        all_data = source_metadata+','+ d8 +','+ d24 +',%s,%s,%s'%(dist_between(rmean8, rmean24)*self.delta*3600, patch_flag, cut_string)      
        all_data +=  ',%s' %(cut_val) #',%s \n' %(cut_val)  
        #print(all_data)
        
        '''pd.Series([data[index-1]['YB'], data[index-1]['l'], data[index-1]['b'], data[index-1]['r'], 
                      totalresid8, residratio8, totalresid24, residratio24, 
                      dist_between(rmean8, rmean24)*self.delta*3600, patch_flag, cut_string,
                      cut_val])'''
        '''YB ID, l, b, User Radius, 
        8um Amplitude (MJy/Sr),D[8um Amplitude], 
        8um x mean (pix), D[8um x mean ], 8um y mean (pix), D[8um y mean] ,8um dist from center (pix), 8um x sd dev (arcsec), D[8um x sd dev ], 8um y st dev (arcsec), D[8um y st dev], 8 um theta, D[8 um theta],\
        8um residual (Jy), 8um residual/input flux (ratio), 24um amplitude (MJy/Sr), D[24um amplitude, 24um x mean (pix), D[24um x mean], 24um y mean (pix), D[24um y mean], 
        24um Dist from center (pix),24um x sd dev (arcsec), D[24um x sd dev], 24um y st dev (arcsec), D[24um y st dev ], 24um theta, D[24um theta], 24um residual (Jy), 24um residual/input flux ratio, \
        dist between means (arcsec), Patch Flag, classification string, cut value\n'''
        
        data_list =  [x.strip() for x in all_data.split(',')]
        for i in range(0, len(data_list)-2):
            data_list[i] = float(data_list[i])
        data_list[-1] = float(data_list[-1])

        df.loc[data[index-1]['YB']] = data_list
        #f.write(all_data) # dump everything to the spreadsheet
        #print(f)
        #}
        
        
        
        if not make_plots:
            return 0
        
        #this is where the plotting happens
        #{
        r = np.zeros(image8.shape)
        g = np.zeros(image8.shape)
        b = np.zeros(image8.shape)
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
        #ax.imshow(make_lupton_rgb(r, g, b, stretch=200, Q=0), norm=LogNorm())
        #ax.imshow(make_lupton_rgb(r, g, b, stretch=200, Q=0))
        ax.imshow(make_lupton_rgb(r, g, b, stretch=75, Q=0))
        ax.set_xlabel('')

        
        #rg residuals
        ax2 = plt.subplot(row, column,2, title = 'residuals', projection = self.w)
        r = resid24
        g = resid8
        #ax2.set_xlabel('')
        ax2.imshow(make_lupton_rgb(r, g, b, stretch=75, Q=0))
        ax2.add_patch(ell8w)
        ax2.add_patch(ell24w)
        ax2.add_patch(usr_circlew)
        #plot of bad pixels and handy info
        #12/22/22: Removing get_category from ax3
        #ax3 = plt.subplot(row, column, 3, title = 'Patch flag:' + str(patch_flag)\
        #                  +'\n Category: '+ get_category(index), projection = self.w)    
        #ax3 = plt.subplot(row, column, 3, title = 'Patch flag:' + str(patch_flag),\
        #                 projection = self.w)
        #ax3.imshow(bad_pixel_map)
      
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
            ax4.add_patch(tge)

            
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
        plt.tight_layout()
        #print(cut_val, cut_string)
        cuts_count[index] = cut_val
        
        if sort_into_cut_folders:
            cut_folder = 'Cut%s' % cut_val
            if not os.path.exists(output_directory + cut_folder): #Checks if the cut folder exists, makes a new one if not
                #print( cut_folder)
                os.makedirs(output_directory + cut_folder)
        else: cut_folder = ''

        
        fig.suptitle(make_title(index), fontsize=20, y=0.99)
        fig.savefig(output_directory + cut_folder +'/'+ make_title(index) + '.png') #Dump the picture to an image file
        plt.close()
        
        
        
        #}
        
        

        
        
        
        

        
        
        
        

        

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

#class that obtains the images necessary for the 2D-gaussian to reconstruct FITS cutouts from user coordinates.
#update_mosaic loads the mosaics from your computer, whilst update_mosaic_api loads it from Google Drive
class choose_image():
    def __init__(self):
        self.path8 = None
        self.um8 = None
        self.um8data = None
        self.um8w = None
        self.path12 = None
        self.um12 = None
        self.um12data = None 
        self.um12w = None
        self.path24 = None
        self.um24 = None
        self.um24data = None
        self.um24w = None
        self.um70 = None
        self.um70data = None 
        self.um70w = None  
        
    def update_mosaic_api(self, l, b):
        #path = '.'
        #currently don't need the WCS files for 12, 24 um because they
        #are reprojections onto 8um coordinate grid
        #GWC added 'b' on 4/5/22. 
        #Adding mosaics 021, 024, 027 on 10/17/23.
        if l > 19.5 and l <= 22.5:
            path8 = 'GLM_02100+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_02100_mosaic.fits'
            path24 = 'MIPSGAL_24um_02100_mosaic.fits'
            path70 = 'PACS_70um_02100_mosaic.fits'
        elif l > 22.5 and l <= 25.5:
            path8 = 'GLM_02400+0000_mosaic_I4.fits'
            path12 ='WISE_12um_02400_mosaic.fits'
            path24 = 'MIPSGAL_24um_02400_mosaic.fits'
            path70 = 'PACS_70um_02400_mosaic.fits'
        elif l > 25.5 and l <= 28.5:
            path8 = 'GLM_02700+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_02700_mosaic.fits'
            path24 = 'MIPSGAL_24um_02700_mosaic.fits'
            path70 ='PACS_70um_02700_mosaic.fits'
        elif l > 28.5 and l <= 31.5:
            path8 = 'GLM_03000+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_03000_mosaic.fits'
            path24 = 'MIPSGAL_03000_mosaic_reprojected.fits'
            path70 = 'PACS_70um_03000_mosaic.fits'
        elif l > 31.5 and l <= 34.5:
            path8 = 'GLM_03300+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_03300_mosaic.fits'
            path24 = 'MIPSGAL_03300_mosaic_reprojected.fits'
            path70 = 'PACS_70um_03300_mosaic.fits'
        elif l > 34.5 and l <= 37.5:
            path8 = 'GLM_03600+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_03600_mosaic.fits'
            path24 = 'MIPSGAL_03600_mosaic_reprojected.fits'
            path70 = 'PACS_70um_03600_mosaic.fits'
        elif l > 37.5 and l <= 40.5:
            path8 = 'GLM_03900+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_03900_mosaic.fits'
            path24 = 'MIPSGAL_03900_mosaic_reprojected.fits'
            path70 = 'PACS_70um_03900_mosaic.fits'
        elif l > 40.5 and l <= 43.5:
            path8 = 'GLM_04200+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_04200_mosaic.fits'
            path24 = 'MIPSGAL_24um_04200_mosaic.fits'
            path70 = 'PACS_70um_04200_mosaic.fits'
        elif l > 43.5 and l <= 46.5:
            path8 = 'GLM_04500+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_04500_mosaic.fits'
            path24 = 'MIPSGAL_24um_04500_mosaic.fits'
            path70 = 'PACS_70um_04500_mosaic.fits'       
        elif l > 46.5 and l <= 49.5:
            path8 = 'GLM_04800+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_04800_mosaic.fits'
            path24 = 'MIPSGAL_24um_04800_mosaic.fits'
            path70 = 'PACS_70um_04800_mosaic.fits'
        elif l > 49.5 and l <= 52.5:
            path8 = 'GLM_05100+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_05100_mosaic.fits'
            path24 = 'MIPSGAL_24um_05100_mosaic.fits'
            path70 = 'PACS_70um_05100_mosaic.fits'
        elif l > 52.5 and l <= 55.5:  
            path8 = 'GLM_05400+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_05400_mosaic.fits'
            path24 = 'MIPSGAL_24um_05400_mosaic.fits'
            path70 = 'PACS_70um_05400_mosaic.fits'
        elif l > 55.5 and l <= 58.5:
            path8 = 'GLM_05700+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_05700_mosaic.fits'
            path24 = 'MIPSGAL_24um_05700_mosaic.fits'
            path70 = 'PACS_70um_05700_mosaic.fits' 
        elif l > 58.5 and l <= 61.5:
            path8 = 'GLM_06000+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_06000_mosaic.fits'
            path24 = 'MIPSGAL_24um_06000_mosaic.fits'
            path70 = 'PACS_70um_06000_mosaic.fits'
        elif l > 61.5 and l <= 64.5:
            path8 = 'GLM_06300+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_06300_mosaic.fits'
            path24 = 'MIPSGAL_24um_06300_mosaic.fits'
            path70 = 'PACS_70um_06300_mosaic.fits'                 
        elif l > 64.5 and l <= 65.5:
            path8 = 'GLM_06600+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_06600_mosaic.fits'
            path24 = 'MIPSGAL_24um_06600_mosaic.fits'
            path70 = 'PACS_70um_06600_mosaic.fits'   
        elif l > 294.8 and l <= 295.5:
            path8 = 'GLM_29400+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_29400_mosaic.fits'
            path24 = 'MIPSGAL_24um_29400_mosaic.fits'
            path70 = 'PACS_70um_29400_mosaic.fits'            
        elif l > 295.5 and l <= 298.5:
            path8 = 'GLM_29700+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_29700_mosaic.fits'
            path24 = 'MIPSGAL_24um_29700_mosaic.fits'
            path70 = 'PACS_70um_29700_mosaic.fits'
        elif l > 298.5 and l <= 301.5:
            path8 = 'GLM_30000+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_30000_mosaic.fits'
            path24 = 'MIPSGAL_24um_30000_mosaic.fits'
            path70 = 'PACS_70um_30000_mosaic.fits'
        elif l > 301.5 and l <= 304.5:
            path8 = 'GLM_30300+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_30300_mosaic.fits'
            path24 = 'MIPSGAL_24um_30300_mosaic.fits'
            path70 = 'PACS_70um_30300_mosaic.fits'
        elif l > 304.5 and l <= 307.5:
            path8 = 'GLM_30600+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_30600_mosaic.fits'
            path24 = 'MIPSGAL_24um_30600_mosaic.fits'
            path70 = 'PACS_70um_30600_mosaic.fits'     
        #Added these mosaics on 7/11/23. For some reason, many of the elif statements
        #for regions I've done are not here. I added these above on 7/26/23. I had to
        #copy them over from ExpertPhotom.py
        #Adding more as I complete photometry for each sector.
        elif l > 307.5 and l <= 310.5:
            path8 = 'GLM_30900+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_30900_mosaic.fits'
            path24 = 'MIPSGAL_24um_30900_mosaic.fits'
            path70 = 'PACS_70um_30900_mosaic.fits'
        elif l > 310.5 and l <= 313.5:
            path8 = 'GLM_31200+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_31200_mosaic.fits'
            path24 = 'MIPSGAL_24um_31200_mosaic.fits'
            path70 = 'PACS_70um_31200_mosaic.fits'
        elif l > 313.5 and l <= 316.5:
            path8 = 'GLM_31500+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_31500_mosaic.fits'
            path24 = 'MIPSGAL_24um_31500_mosaic.fits'
            path70 = 'PACS_70um_31500_mosaic.fits'  
        elif l > 316.5 and l <= 319.5:
            path8 = 'GLM_31800+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_31800_mosaic.fits'
            path24 = 'MIPSGAL_24um_31800_mosaic.fits'
            path70 = 'PACS_70um_31800_mosaic.fits'    
        elif l > 319.5 and l <= 322.5:
            path8 = 'GLM_32100+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_32100_mosaic.fits'
            path24 = 'MIPSGAL_24um_32100_mosaic.fits'
            path70 = 'PACS_70um_32100_mosaic.fits' 
        elif l > 322.5 and l <= 325.5:
            path8 = 'GLM_32400+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_32400_mosaic.fits'
            path24 = 'MIPSGAL_24um_32400_mosaic.fits'
            path70 = 'PACS_70um_32400_mosaic.fits'        
        elif l > 325.5 and l <= 328.5:
            path8 = 'GLM_32700+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_32700_mosaic.fits'
            path24 = 'MIPSGAL_24um_32700_mosaic.fits'
            path70 = 'PACS_70um_32700_mosaic.fits'       
        elif l > 328.5 and l <= 331.5:
            path8 = 'GLM_33000+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_33000_mosaic.fits'
            path24 = 'MIPSGAL_24um_33000_mosaic.fits'
            path70 = 'PACS_70um_33000_mosaic.fits' 
        elif l > 331.5 and l <= 334.5:
            path8 = 'GLM_33300+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_33300_mosaic.fits'
            path24 = 'MIPSGAL_24um_33300_mosaic.fits'
            path70 = 'PACS_70um_33300_mosaic.fits'         
        elif l > 334.5 and l <= 337.5:
            path8 = 'GLM_33600+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_33600_mosaic.fits'
            path24 = 'MIPSGAL_24um_33600_mosaic.fits'
            path70 = 'PACS_70um_33600_mosaic.fits'
        elif l > 337.5 and l <= 340.5:
            path8 = 'GLM_33900+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_33900_mosaic.fits'
            path24 = 'MIPSGAL_24um_33900_mosaic.fits'
            path70 = 'PACS_70um_33900_mosaic.fits'      
        elif l > 340.5 and l <= 343.5:
            path8 = 'GLM_34200+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_34200_mosaic.fits'
            path24 = 'MIPSGAL_24um_34200_mosaic.fits'
            path70 = 'PACS_70um_34200_mosaic.fits'
        elif l > 343.5 and l <= 346.5:
            path8 = 'GLM_34500+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_34500_mosaic.fits'
            path24 = 'MIPSGAL_24um_34500_mosaic.fits'
            path70 = 'PACS_70um_34500_mosaic.fits'
        elif l > 346.5 and l <= 349.5:
            path8 = 'GLM_34800+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_34800_mosaic.fits'
            path24 = 'MIPSGAL_24um_34800_mosaic.fits'
            path70 = 'PACS_70um_34800_mosaic.fits'
        elif l > 349.5 and l <= 352.5:
            path8 = 'GLM_35100+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_35100_mosaic.fits'
            path24 = 'MIPSGAL_24um_35100_mosaic.fits'
            path70 = 'PACS_70um_35100_mosaic.fits'
        elif l > 352.5 and l <= 355.5:
            path8 = 'GLM_35400+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_35400_mosaic.fits'
            path24 = 'MIPSGAL_24um_35400_mosaic.fits'
            path70 = 'PACS_70um_35400_mosaic.fits'
        elif l > 355.5 and l <= 358.5:
            path8 = 'GLM_35700+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_35700_mosaic.fits'
            path24 = 'MIPSGAL_24um_35700_mosaic.fits'
            path70 = 'PACS_70um_35700_mosaic.fits'  
        elif (l > 358.5 and l <= 360.1) or (l > -0.1 and l <= 1.5):
            path8 = 'GLM_00000+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_00000_mosaic.fits'
            path24 = 'MIPSGAL_24um_00000_mosaic.fits'
            path70 = 'PACS_70um_00000_mosaic.fits'
        elif l > 1.5 and l <= 4.5:
            path8 = 'GLM_00300+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_00300_mosaic.fits'
            path24 = 'MIPSGAL_24um_00300_mosaic.fits'
            path70 = 'PACS_70um_00300_mosaic.fits'
        elif l > 4.5 and l <= 7.5:
            path8 = 'GLM_00600+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_00600_mosaic.fits'
            path24 = 'MIPSGAL_24um_00600_mosaic.fits'
            path70 = 'PACS_70um_00600_mosaic.fits'
        elif l > 7.5 and l <= 10.5:
            path8 = 'GLM_00900+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_00900_mosaic.fits'
            path24 = 'MIPSGAL_24um_00900_mosaic.fits'
            path70 = 'PACS_70um_00900_mosaic.fits'
        elif l > 10.5 and l <= 13.5:
            path8 = 'GLM_01200+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_01200_mosaic.fits'
            path24 = 'MIPSGAL_24um_01200_mosaic.fits'
            path70 = 'PACS_70um_01200_mosaic.fits'
        elif l > 13.5 and l <= 16.5:
            path8 = 'GLM_01500+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_01500_mosaic.fits'
            path24 = 'MIPSGAL_24um_01500_mosaic.fits'
            path70 = 'PACS_70um_01500_mosaic.fits'
        elif l > 16.5 and l <= 19.5:
            path8 = 'GLM_01800+0000_mosaic_I4.fits'
            path12 = 'WISE_12um_01800_mosaic.fits'
            path24 = 'MIPSGAL_24um_01800_mosaic.fits'
            path70 = 'PACS_70um_01800_mosaic.fits'
        #The following are for the SMOG region.  
        #GWC: Something went wonky on 2/7/24 -- need to revisit how to cover SMOG.
        elif l > 101.0 and l <= 105.59 and b < 3.06:
            path8 = 'SMOG_08um_10300_mosaic.fits'
            path12 = 'SMOG_12um_10300_mosaic.fits'
            path24 = 'SMOG_24um_10300_mosaic.fits'
            path70 = 'SMOG_PACS_70um_10300_mosaic.fits'
            # Replaced 'mosaics/SMOG_70um_10300_mosaic.fits') with PACS on 7/7/23
        elif l > 101.0 and l <= 105.59 and b >= 3.06:
            path8 = 'SMOG_08um_10300_mosaic.fits'
            path12 = 'SMOG_12um_10300_mosaic.fits'
            path24 = 'SMOG_24um_10300_mosaic_high_b.fits'
            path70 = 'SMOG_PACS_70um_10300_mosaic.fits'
        elif l > 105.59 and l <= 110.2:
            path8 = 'SMOG_08um_10700_mosaic.fits'
            path12 = 'SMOG_12um_10700_mosaic.fits'
            path24 = 'SMOG_24um_10700_mosaic.fits'
            path70 = 'SMOG_PACS_70um_10700_mosaic.fits'
            # Replaced 'mosaics/SMOG_70um_10700_mosaic.fits') with PACS on 7/7/23
        # The following were added for Cyg-X by GW-C on 2/7/24.
        elif l > 75.5 and l <= 76.5:
            path8 = 'CYGX_08um_07500+0050_mosaic.fits'
            path12 = 'CYGX_12um_07500+0050_mosaic.fits'
            path24 = 'CYGX_24um_07500+0050_mosaic.fits'
            path70 = 'CYGX_70um_07500+0050_mosaic.fits'
        elif l > 76.5 and l <= 79.5 and b < 0.82:
            path8 = 'CYGX_08um_07800-0085_mosaic.fits'
            path12 = 'CYGX_12um_07800-0085_mosaic.fits'
            path24 = 'CYGX_24um_07800-0085_mosaic.fits'
            path70 = 'CYGX_70um_07800-0085_mosaic.fits'
        elif l > 76.5 and l <= 79.5 and b >= 0.82:
            path8 = 'CYGX_08um_07800+0250_mosaic.fits'
            path12 = 'CYGX_12um_07800+0250_mosaic.fits'
            path24 = 'CYGX_24um_07800+0250_mosaic.fits'
            path70 = 'CYGX_70um_07800+0250_mosaic.fits'
        elif l > 79.5 and l <= 82.5 and b < 0.82:
            path8 = 'CYGX_08um_08100-0070_mosaic.fits'
            path12 = 'CYGX_12um_08100-0070_mosaic.fits'
            path24 = 'CYGX_24um_08100-0070_mosaic.fits'
            path70 = 'CYGX_70um_08100-0070_mosaic.fits'
        elif l > 79.5 and l <= 82.5 and b >= 0.82:
            path8 = 'CYGX_08um_08100+0235_mosaic.fits'
            path12 = 'CYGX_12um_08100+0235_mosaic.fits'
            path24 = 'CYGX_24um_08100+0235_mosaic.fits'
            path70 = 'CYGX_70um_08100+0235_mosaic.fits'
        elif l > 82.5 and l <= 83.0:
            path8 = 'CYGX_08um_08400+0005_mosaic.fits'
            path12 = 'CYGX_12um_08400+0005_mosaic.fits'
            path24 = 'CYGX_24um_08400+0005_mosaic.fits'
            path70 = 'CYGX_70um_08400+0005_mosaic.fits'
        else:
            # GWC revised print statement from "outside the pilot..."
            print('Your YB is outside the region.')
            print('Please try again.')
            sys.exit()



        if(self.path8 == path8 and self.path12 == path12 and self.path24==path24 and self.path70 ==path70):
            return 0


        
        file_list = drive.ListFile({'q': f"title='{path8}' and trashed=false"}).GetList()
        if not file_list:
            print(f"File '{path8}' not found in Google Drive.")
        file_obj = file_list[0]
        file_obj.GetContentFile(temp_path8)
        temp = fits.open(temp_path8)[0]
        self.um8 = temp#.copy()
        self.um8data = temp.data#.copy()
        self.um8w = wcs.WCS(temp.header)#.copy()
        self.path8 = path8
        
        
        file_list = drive.ListFile({'q': f"title='{path12}' and trashed=false"}).GetList()
        if not file_list:
            print(f"File '{path12}' not found in Google Drive.")
        file_obj = file_list[0]
        file_obj.GetContentFile(temp_path12)
        temp = fits.open(temp_path12)[0]
        self.um12 = temp#.copy()
        self.um12data = temp.data#.copy()
        self.um12w = wcs.WCS(temp.header)#.copy()
        self.path12 = path12
        
        file_list = drive.ListFile({'q': f"title='{path24}' and trashed=false"}).GetList()
        if not file_list:
            print(f"File '{path24}' not found in Google Drive.")
        file_obj = file_list[0]
        file_obj.GetContentFile(temp_path24)
        temp = fits.open(temp_path24)[0]
        self.um24 = temp#.copy()
        self.um24data = temp.data#.copy()
        self.um24w = wcs.WCS(temp.header)#.copy()
        self.path24 = path24
        
        file_list = drive.ListFile({'q': f"title='{path70}' and trashed=false"}).GetList()
        if not file_list:
            print(f"File '{path70}' not found in Google Drive.")
        file_obj = file_list[0]
        file_obj.GetContentFile(temp_path70)
        temp = fits.open(temp_path70)[0]
        self.um70 = temp#.copy()
        self.um70data = temp.data#.copy()
        self.um70w = wcs.WCS(temp.header)#.copy()
        self.path70 = path70
        
        #os.remove(temp_path)
        
    def update_mosaic(self, l, b):
        if l > 1.5 and l <= 4.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_00300+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_00300_mosaic.fits')
            path24 = os.path.join(
                                   path,
                                      'mosaics/MIPSGAL_24um_00300_mosaic.fits')
            path70 = os.path.join(
                                   path,
                                       'mosaics/PACS_70um_00300_mosaic.fits')
        elif l > 4.5 and l <= 7.5:
            path8 = os.path.join(path,
                                  'mosaics/GLM_00600+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                   'mosaics/WISE_12um_00600_mosaic.fits')
            path24 = os.path.join(
                                    path,
                                       'mosaics/MIPSGAL_24um_00600_mosaic.fits')
            path70 = os.path.join(
                                    path,
                                        'mosaics/PACS_70um_00600_mosaic.fits')
        elif l > 7.5 and l <= 10.5:
            path8 = os.path.join(path,
                                  'mosaics/GLM_00900+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                   'mosaics/WISE_12um_00900_mosaic.fits')
            path24 = os.path.join(
                                    path,
                                       'mosaics/MIPSGAL_24um_00900_mosaic.fits')
            path70 = os.path.join(
                                    path,
                                        'mosaics/PACS_70um_00900_mosaic.fits')
        elif l > 10.5 and l <= 13.5:
            path8 = os.path.join(path,
                                  'mosaics/GLM_01200+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                   'mosaics/WISE_12um_01200_mosaic.fits')
            path24 = os.path.join(
                                    path,
                                       'mosaics/MIPSGAL_24um_01200_mosaic.fits')
            path70 = os.path.join(
                                    path,
                                        'mosaics/PACS_70um_01200_mosaic.fits')
        elif l > 13.5 and l <= 16.5:
            path8 = os.path.join(path,
                                  'mosaics/GLM_01500+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                   'mosaics/WISE_12um_01500_mosaic.fits')
            path24 = os.path.join(
                                    path,
                                       'mosaics/MIPSGAL_24um_01500_mosaic.fits')
            path70 = os.path.join(
                                    path,
                                        'mosaics/PACS_70um_01500_mosaic.fits')
        elif l > 16.5 and l <= 19.5:
            path8 = os.path.join(path,
                                  'mosaics/GLM_01800+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                   'mosaics/WISE_12um_01800_mosaic.fits')
            path24 = os.path.join(
                                    path,
                                       'mosaics/MIPSGAL_24um_01800_mosaic.fits')
            path70 = os.path.join(
                                    path,
                                        'mosaics/PACS_70um_01800_mosaic.fits')
        #Adding mosaics 021, 024, 027 on 10/17/23.
        elif l > 19.5 and l <= 22.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_02100+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_02100_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/MIPSGAL_24um_02100_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/PACS_70um_02100_mosaic.fits')    
        elif l > 22.5 and l <= 25.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_02400+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_02400_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/MIPSGAL_24um_02400_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/PACS_70um_02400_mosaic.fits')    
        elif l > 25.5 and l <= 28.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_02700+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_02700_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/MIPSGAL_24um_02700_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/PACS_70um_02700_mosaic.fits')    
        elif l > 28.5 and l <= 31.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_03000+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_03000_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/MIPSGAL_03000_mosaic_reprojected.fits')
            path70 = os.path.join(
                path,
                'mosaics/PACS_70um_03000_mosaic.fits')
        elif l > 31.5 and l <= 34.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_03300+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_03300_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/MIPSGAL_03300_mosaic_reprojected.fits')
            path70 = os.path.join(
                path,
                'mosaics/PACS_70um_03300_mosaic.fits')
        elif l > 34.5 and l <= 37.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_03600+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_03600_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/MIPSGAL_03600_mosaic_reprojected.fits')
            path70 = os.path.join(
                path,
                'mosaics/PACS_70um_03600_mosaic.fits')
        elif l > 37.5 and l <= 40.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_03900+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_03900_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/MIPSGAL_03900_mosaic_reprojected.fits')
            path70 = os.path.join(
                path,
                'mosaics/PACS_70um_03900_mosaic.fits')
        elif l > 40.5 and l <= 43.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_04200+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_04200_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/MIPSGAL_24um_04200_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/PACS_70um_04200_mosaic.fits')
        elif l > 43.5 and l <= 46.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_04500+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_04500_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/MIPSGAL_24um_04500_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/PACS_70um_04500_mosaic.fits')       
        elif l > 46.5 and l <= 49.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_04800+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_04800_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/MIPSGAL_24um_04800_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/PACS_70um_04800_mosaic.fits') 
        elif l > 49.5 and l <= 52.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_05100+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_05100_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/MIPSGAL_24um_05100_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/PACS_70um_05100_mosaic.fits')
        elif l > 52.5 and l <= 55.5:  
            path8 = os.path.join(path,
                                 'mosaics/GLM_05400+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_05400_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/MIPSGAL_24um_05400_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/PACS_70um_05400_mosaic.fits')
        elif l > 55.5 and l <= 58.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_05700+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_05700_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/MIPSGAL_24um_05700_mosaic.fits')
            path70 = os.path.join(
                    path,
                    'mosaics/PACS_70um_05700_mosaic.fits')   
        elif l > 58.5 and l <= 61.5:
            path8 = os.path.join(path,
                                     'mosaics/GLM_06000+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                      'mosaics/WISE_12um_06000_mosaic.fits')
            path24 = os.path.join(
                    path,
                    'mosaics/MIPSGAL_24um_06000_mosaic.fits')
            path70 = os.path.join(
                    path,
                    'mosaics/PACS_70um_06000_mosaic.fits')  
        elif l > 61.5 and l <= 64.5:
            path8 = os.path.join(path,
                                     'mosaics/GLM_06300+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                      'mosaics/WISE_12um_06300_mosaic.fits')
            path24 = os.path.join(
                    path,
                    'mosaics/MIPSGAL_24um_06300_mosaic.fits')
            path70 = os.path.join(
                    path,
                    'mosaics/PACS_70um_06300_mosaic.fits')                   
        elif l > 64.5 and l <= 65.5:
            path8 = os.path.join(path,
                                     'mosaics/GLM_06600+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                      'mosaics/WISE_12um_06600_mosaic.fits')
            path24 = os.path.join(
                    path,
                    'mosaics/MIPSGAL_24um_06600_mosaic.fits')
            path70 = os.path.join(
                    path,
                    'mosaics/PACS_70um_06600_mosaic.fits') 

        # The following were added for Cyg-X by GW-C on 2/7/24.
        elif l > 75.5 and l <= 76.5:
            path8 = os.path.join(path,
                                 'mosaics/CYGX_08um_07500+0050_mosaic.fits')
            path12 = os.path.join(path,
                                  'mosaics/CYGX_12um_07500+0050_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/CYGX_24um_07500+0050_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/CYGX_70um_07500+0050_mosaic.fits')
        elif l > 76.5 and l <= 79.5 and b < 0.82:
            path8 = os.path.join(path,
                                 'mosaics/CYGX_08um_07800-0085_mosaic.fits')
            path12 = os.path.join(path,
                                  'mosaics/CYGX_12um_07800-0085_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/CYGX_24um_07800-0085_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/CYGX_70um_07800-0085_mosaic.fits')
        elif l > 76.5 and l <= 79.5 and b >= 0.82:
            path8 = os.path.join(path,
                                 'mosaics/CYGX_08um_07800+0250_mosaic.fits')
            path12 = os.path.join(path,
                                  'mosaics/CYGX_12um_07800+0250_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/CYGX_24um_07800+0250_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/CYGX_70um_07800+0250_mosaic.fits')
        elif l > 79.5 and l <= 82.5 and b < 0.82:
            path8 = os.path.join(path,
                                 'mosaics/CYGX_08um_08100-0070_mosaic.fits')
            path12 = os.path.join(path,
                                  'mosaics/CYGX_12um_08100-0070_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/CYGX_24um_08100-0070_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/CYGX_70um_08100-0070_mosaic.fits')
        elif l > 79.5 and l <= 82.5 and b >= 0.82:
            path8 = os.path.join(path,
                                 'mosaics/CYGX_08um_08100+0235_mosaic.fits')
            path12 = os.path.join(path,
                                  'mosaics/CYGX_12um_08100+0235_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/CYGX_24um_08100+0235_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/CYGX_70um_08100+0235_mosaic.fits')
        elif l > 82.5 and l <= 83.0:
            path8 = os.path.join(path,
                                 'mosaics/CYGX_08um_08400+0005_mosaic.fits')
            path12 = os.path.join(path,
                                  'mosaics/CYGX_12um_08400+0005_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/CYGX_24um_08400+0005_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/CYGX_70um_08400+0005_mosaic.fits')

        #The following are for the SMOG region.  
        #GWC: Something went wonky on 2/7/24 -- need to revisit how to cover SMOG.
        elif l > 101.0 and l <= 105.59 and b < 3.06:
            path8 = os.path.join(path,
                                 'mosaics/SMOG_08um_10300_mosaic.fits')
            path12 = os.path.join(path,
                                  'mosaics/SMOG_12um_10300_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/SMOG_24um_10300_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/SMOG_PACS_70um_10300_mosaic.fits')
            # Replaced 'mosaics/SMOG_70um_10300_mosaic.fits') with PACS on 7/7/23
        elif l > 101.0 and l <= 105.59 and b >= 3.06:
            path8 = os.path.join(path,
                                 'mosaics/SMOG_08um_10300_mosaic.fits')
            path12 = os.path.join(path,
                                  'mosaics/SMOG_12um_10300_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/SMOG_24um_10300_mosaic_high_b.fits')
            path70 = os.path.join(
                path,
                'mosaics/SMOG_PACS_70um_10300_mosaic.fits')
        elif l > 105.59 and l <= 110.2:
            path8 = os.path.join(path,
                                 'mosaics/SMOG_08um_10700_mosaic.fits')
            path12 = os.path.join(path,
                                  'mosaics/SMOG_12um_10700_mosaic.fits')
            path24 = os.path.join(
                path,
                'mosaics/SMOG_24um_10700_mosaic.fits')
            path70 = os.path.join(
                path,
                'mosaics/SMOG_PACS_70um_10700_mosaic.fits')
            # Replaced 'mosaics/SMOG_70um_10700_mosaic.fits') with PACS on 7/7/23
        
        elif l > 294.8 and l <= 295.5:
            path8 = os.path.join(path,
                                     'mosaics/GLM_29400+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                      'mosaics/WISE_12um_29400_mosaic.fits')
            path24 = os.path.join(
                    path,
                    'mosaics/MIPSGAL_24um_29400_mosaic.fits')
            path70 = os.path.join(
                    path,
                    'mosaics/PACS_70um_29400_mosaic.fits')              
        elif l > 295.5 and l <= 298.5:
            path8 = os.path.join(path,
                                     'mosaics/GLM_29700+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                      'mosaics/WISE_12um_29700_mosaic.fits')
            path24 = os.path.join(
                    path,
                    'mosaics/MIPSGAL_24um_29700_mosaic.fits')
            path70 = os.path.join(
                    path,
                    'mosaics/PACS_70um_29700_mosaic.fits')
        elif l > 298.5 and l <= 301.5:
            path8 = os.path.join(path,
                                     'mosaics/GLM_30000+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                      'mosaics/WISE_12um_30000_mosaic.fits')
            path24 = os.path.join(
                    path,
                    'mosaics/MIPSGAL_24um_30000_mosaic.fits')
            path70 = os.path.join(
                    path,
                    'mosaics/PACS_70um_30000_mosaic.fits')
        elif l > 301.5 and l <= 304.5:
            path8 = os.path.join(path,
                                     'mosaics/GLM_30300+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                      'mosaics/WISE_12um_30300_mosaic.fits')
            path24 = os.path.join(
                    path,
                    'mosaics/MIPSGAL_24um_30300_mosaic.fits')
            path70 = os.path.join(
                    path,
                    'mosaics/PACS_70um_30300_mosaic.fits')
        elif l > 304.5 and l <= 307.5:
            path8 = os.path.join(path,
                                     'mosaics/GLM_30600+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                      'mosaics/WISE_12um_30600_mosaic.fits')
            path24 = os.path.join(
                    path,
                    'mosaics/MIPSGAL_24um_30600_mosaic.fits')
            path70 = os.path.join(
                    path,
                    'mosaics/PACS_70um_30600_mosaic.fits')       
        #Added these mosaics on 7/11/23. For some reason, many of the elif statements
        #for regions I've done are not here. I added these above on 7/26/23. I had to
        #copy them over from ExpertPhotom.py
        #Adding more as I complete photometry for eacj sector.
        elif l > 307.5 and l <= 310.5:
            path8 = os.path.join(path,
                                     'mosaics/GLM_30900+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                      'mosaics/WISE_12um_30900_mosaic.fits')
            path24 = os.path.join(
                    path,
                    'mosaics/MIPSGAL_24um_30900_mosaic.fits')
            path70 = os.path.join(
                    path,
                    'mosaics/PACS_70um_30900_mosaic.fits')
        elif l > 310.5 and l <= 313.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_31200+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_31200_mosaic.fits')
            path24 = os.path.join(
                        path,
                        'mosaics/MIPSGAL_24um_31200_mosaic.fits')
            path70 = os.path.join(
                        path,
                        'mosaics/PACS_70um_31200_mosaic.fits')
        elif l > 313.5 and l <= 316.5:
            path8 = os.path.join(path,
                                  'mosaics/GLM_31500+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                   'mosaics/WISE_12um_31500_mosaic.fits')
            path24 = os.path.join(
                         path,
                         'mosaics/MIPSGAL_24um_31500_mosaic.fits')
            path70 = os.path.join(
                         path,
                         'mosaics/PACS_70um_31500_mosaic.fits')    
        elif l > 316.5 and l <= 319.5:
            path8 = os.path.join(path,
                                   'mosaics/GLM_31800+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                    'mosaics/WISE_12um_31800_mosaic.fits')
            path24 = os.path.join(
                          path,
                          'mosaics/MIPSGAL_24um_31800_mosaic.fits')
            path70 = os.path.join(
                          path,
                          'mosaics/PACS_70um_31800_mosaic.fits')      
        elif l > 319.5 and l <= 322.5:
            path8 = os.path.join(path,
                                   'mosaics/GLM_32100+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                    'mosaics/WISE_12um_32100_mosaic.fits')
            path24 = os.path.join(
                          path,
                          'mosaics/MIPSGAL_24um_32100_mosaic.fits')
            path70 = os.path.join(
                          path,
                          'mosaics/PACS_70um_32100_mosaic.fits')   
        elif l > 322.5 and l <= 325.5:
            path8 = os.path.join(path,
                                   'mosaics/GLM_32400+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                    'mosaics/WISE_12um_32400_mosaic.fits')
            path24 = os.path.join(
                          path,
                          'mosaics/MIPSGAL_24um_32400_mosaic.fits')
            path70 = os.path.join(
                          path,
                          'mosaics/PACS_70um_32400_mosaic.fits')          
        elif l > 325.5 and l <= 328.5:
            path8 = os.path.join(path,
                                   'mosaics/GLM_32700+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                    'mosaics/WISE_12um_32700_mosaic.fits')
            path24 = os.path.join(
                          path,
                          'mosaics/MIPSGAL_24um_32700_mosaic.fits')
            path70 = os.path.join(
                          path,
                          'mosaics/PACS_70um_32700_mosaic.fits')         
        elif l > 328.5 and l <= 331.5:
            path8 = os.path.join(path,
                                       'mosaics/GLM_33000+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                        'mosaics/WISE_12um_33000_mosaic.fits')
            path24 = os.path.join(
                              path,
                              'mosaics/MIPSGAL_24um_33000_mosaic.fits')
            path70 = os.path.join(
                              path,
                              'mosaics/PACS_70um_33000_mosaic.fits')   
        elif l > 331.5 and l <= 334.5:
            path8 = os.path.join(path,
                                       'mosaics/GLM_33300+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                        'mosaics/WISE_12um_33300_mosaic.fits')
            path24 = os.path.join(
                              path,
                              'mosaics/MIPSGAL_24um_33300_mosaic.fits')
            path70 = os.path.join(
                              path,
                              'mosaics/PACS_70um_33300_mosaic.fits')           
        elif l > 334.5 and l <= 337.5:
            path8 = os.path.join(path,
                                    'mosaics/GLM_33600+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_33600_mosaic.fits')
            path24 = os.path.join(
                                  path,
                                  'mosaics/MIPSGAL_24um_33600_mosaic.fits')
            path70 = os.path.join(
                                  path,
                                  'mosaics/PACS_70um_33600_mosaic.fits')  
        elif l > 337.5 and l <= 340.5:
            path8 = os.path.join(path,
                                     'mosaics/GLM_33900+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                   'mosaics/WISE_12um_33900_mosaic.fits')
            path24 = os.path.join(
                                   path,
                                   'mosaics/MIPSGAL_24um_33900_mosaic.fits')
            path70 = os.path.join(
                                   path,
                                   'mosaics/PACS_70um_33900_mosaic.fits')        
        elif l > 340.5 and l <= 343.5:
            path8 = os.path.join(path,
                                     'mosaics/GLM_34200+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                   'mosaics/WISE_12um_34200_mosaic.fits')
            path24 = os.path.join(
                                   path,
                                   'mosaics/MIPSGAL_24um_34200_mosaic.fits')
            path70 = os.path.join(
                                   path,
                                   'mosaics/PACS_70um_34200_mosaic.fits') 
        elif l > 343.5 and l <= 346.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_34500+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_34500_mosaic.fits')
            path24 = os.path.join(
                                   path,
                                      'mosaics/MIPSGAL_24um_34500_mosaic.fits')
            path70 = os.path.join(
                                   path,
                                       'mosaics/PACS_70um_34500_mosaic.fits') 
        elif l > 346.5 and l <= 349.5:
            path8 = os.path.join(path,
                                'mosaics/GLM_34800+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                 'mosaics/WISE_12um_34800_mosaic.fits')
            path24 = os.path.join(
                                  path,
                                     'mosaics/MIPSGAL_24um_34800_mosaic.fits')
            path70 = os.path.join(
                                  path,
                                      'mosaics/PACS_70um_34800_mosaic.fits') 
        elif l > 349.5 and l <= 352.5:
            path8 = os.path.join(path,
                                'mosaics/GLM_35100+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                 'mosaics/WISE_12um_35100_mosaic.fits')
            path24 = os.path.join(
                                  path,
                                     'mosaics/MIPSGAL_24um_35100_mosaic.fits')
            path70 = os.path.join(
                                  path,
                                      'mosaics/PACS_70um_35100_mosaic.fits') 
        elif l > 352.5 and l <= 355.5:
            path8 = os.path.join(path,
                                'mosaics/GLM_35400+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                 'mosaics/WISE_12um_35400_mosaic.fits')
            path24 = os.path.join(
                                  path,
                                     'mosaics/MIPSGAL_24um_35400_mosaic.fits')
            path70 = os.path.join(
                                  path,
                                      'mosaics/PACS_70um_35400_mosaic.fits') 
        elif l > 355.5 and l <= 358.5:
            path8 = os.path.join(path,
                                 'mosaics/GLM_35700+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_35700_mosaic.fits')
            path24 = os.path.join(
                                   path,
                                      'mosaics/MIPSGAL_24um_35700_mosaic.fits')
            path70 = os.path.join(
                                   path,
                                       'mosaics/PACS_70um_35700_mosaic.fits')    
        elif (l > 358.5 and l <= 360.1) or (l > -0.1 and l <= 1.5):
            path8 = os.path.join(path,
                                 'mosaics/GLM_00000+0000_mosaic_I4.fits')
            path12 = os.path.join(path,
                                  'mosaics/WISE_12um_00000_mosaic.fits')
            path24 = os.path.join(
                                   path,
                                      'mosaics/MIPSGAL_24um_00000_mosaic.fits')
            path70 = os.path.join(
                                   path,
                                       'mosaics/PACS_70um_00000_mosaic.fits')
        else:
            # GWC revised print statement from "outside the pilot..."
            print('Your YB is outside the region.')
            print('Please try again.')
            sys.exit()

        temp = fits.open(path8)[0]
        self.um8 = temp
        self.um8data = temp.data
        self.um8w = wcs.WCS(temp.header)
        temp = fits.open(path12)[0]
        self.um12 = temp
        self.um12data = temp.data
        self.um12w = wcs.WCS(temp.header)
        temp = fits.open(path24)[0]
        self.um24 = temp
        self.um24data = temp.data
        self.um24w = wcs.WCS(temp.header)
        temp = fits.open(path70)[0]
        self.um70 = temp
        self.um70data = temp.data
        self.um70w = wcs.WCS(temp.header)
        
        
#class that does the masking and interpolation, returns masked, blanked, interpolated, and residual
class do_interp():
    def __init__(self, img, verts):
        
        #use the clicked values from the user to create a NaN mask
        vertices = np.array([verts], dtype=np.int32)
        xyvals = np.array(verts, dtype=np.int32)
        xmin = min(xyvals[:, 0]) - 5
        xmax = max(xyvals[:, 0]) + 5
        ymin = min(xyvals[:, 1]) - 5
        ymax = max(xyvals[:, 1]) + 5
        #print(xmin, xmax, ymin, ymax)
        mask = np.zeros_like(img)
        inverse_mask = np.zeros_like(img)
        region_mask = np.zeros_like(img)
        cutout = np.zeros_like(img)

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, 255)
        #TURN ALL Non-ZERO to NaN
        inverse_mask[np.nonzero(mask)] = int(
            1)  # ones inside poly, zero outside

        mask[np.nonzero(mask)] = float('nan')
        #TURN ALL ZERO to 1
        mask[np.where(mask == 0)] = int(1)  # nan inside poly, 1 outside
        region_mask = mask
        region_mask = np.nan_to_num(region_mask)  # zero in poly, 1 outside
        cutout[ymin:ymax, xmin:xmax] = mask[ymin:ymax, xmin:xmax]
        #TURN EVERYTHING OUTSIDE THAT RANGE to NaN
        cutout[np.where(cutout == 0)] = float('nan')

        #TAKE image=workask*mask will make a image with original values but NaN in polygon
        #blank = img*mask
        blank = img * cutout
        self.masked = mask
        self.blanked = blank
        goodvals = np.where(np.isfinite(blank))

        #perform the interpolation over the masked coordinates
        x = goodvals[1]  # x values of finite coordinates
        y = goodvals[0]  # y values of finite coordinates

        def get_fvals(x, y):
            range_array = np.arange(x.size)
            vals = np.zeros(x.size)
            for (i, xi, yi) in zip(range_array, x, y):
                vals[i] = img[yi][xi]
            return vals

        fvals = get_fvals(x, y)

        newfunc = interpolate.Rbf(
            x, y, fvals,
            function='multiquadric')  # the function that does interpolation
        allvals = np.where(img)  # whole region to interpolate over
        xnew = allvals[1]
        ynew = allvals[0]
        fnew = newfunc(xnew, ynew)

        #put the interpolated values back into a 2D array for display and other uses
        def make_2D(fnew, xnew, ynew, img):
            new_array = np.zeros(
                (int(xnew.size /
                     ((img.shape)[0])), int(ynew.size / ((img.shape)[1]))),
                dtype=float)
            #print(new_array)
            range_array = np.arange(fnew.size)
            #print("rangearay:",range_array)

            for (i, x, y) in zip(range_array, xnew, ynew):
                new_array[y][x] = fnew[i]

            return new_array

        fnew_2D = make_2D(fnew, xnew, ynew, img)

        self.interp = img * region_mask + fnew_2D * inverse_mask

        #generate the residual image (original - interpolated background)
        self.resid = img - (img * region_mask + fnew_2D * inverse_mask)

#Takes a YBnumber, returns the residuals for the cutouts from the coordinates of the csv file for 24 and 8 in that order
class get_residual():
    def __init__(self, YBnum, Mosaics):
        #path = '.'
        #catalog_name = os.path.join(path, 'USE_THIS_CATALOG_ybcat_MWP_with_ID.csv')
        data = ascii.read(catalog_name, delimiter=',')
        #get the YB's location and radius
        YB_long = data[YBnum-1]['l']
        YB_lat = data[YBnum-1]['b']
        YB_rad = data[YBnum-1]['r']
        #Use the location to determine the correct image files to use
        #GWC added YB_lat on 4/5/22.
        #if use_api:
        #    Mosaics.update_mosaic_api(YB_long, YB_lat)
        #else:
        #    Mosaics.update_mosaic(YB_long, YB_lat)
        try:
            Mosaics.update_mosaic(YB_long, YB_lat)
        except:
            Mosaics.update_mosaic_api(YB_long, YB_lat)


        
        ##          Unit Conversions Info         ##
        # 8 um:
            #GLIMPSE 8 um flux units are MJy/steradian
            #obtain the 8 um pixel scale from the image header
            #this gives the degree/pixel conversion factor used for overdrawing circle
            #and flux conversions (this is x direction only but they are equal in this case)
            #GWC edit 4/1/22: Set pixscale directly to 0.000333333
            #pixscale8 = abs(image.um8w.wcs.cd[0][0])
        pixscale8 = 0.000333333
            #pixscale8 = abs(image.um8w.wcs.cdelt1)
            #print(pixscale8)
            #8 um square degree to square pixel conversion-- x*y
            #GWC edit 4/1/22: Set sqdeg_tosqpix8 directly to pixscale8 * pixscale8
            #sqdeg_tosqpix8 = abs(image.um8w.wcs.cd[0][0]) * abs(
            #    image.um8w.wcs.cd[1][1])
        sqdeg_tosqpix8 = pixscale8 * pixscale8
        #8 um steradian to pixel conversion (used to convert MJy/Pixel)
        #     will be used to convert to MJy
        str_to_pix8 = sqdeg_tosqpix8 * 0.0003046118
        #WISE Units are Data Number per Pixel, Jy/DN is 1.8326*10^-6
        #See http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html
        dn_to_jy12 = 1.83 * 10**-6
        
        #convert YB l and b and radius to pixel coordinates
        ybwcs = np.array([[YB_long, YB_lat]], np.float_)
        pixcoords = Mosaics.um8w.wcs_world2pix(ybwcs, 1)
        YB_long_pix = pixcoords[0][0]
        YB_lat_pix = pixcoords[0][1]
        YB_rad_pix = YB_rad / pixscale8
        
        # This is for the added points to show the user what other YBs are in the images
        # Read in the l, b, and r values for all the YBs and convert them to pixels
        YBloc = pd.read_csv(catalog_name, usecols = ['l', 'b', 'r'])
        # Convert l, b, and r from YBloc into pixels
        for i in range(0, len(YBloc)):
            yblocwcs = np.array([[YBloc['l'][i], YBloc['b'][i]]], np.float_)
            pixcoordsloc = Mosaics.um8w.wcs_world2pix(yblocwcs, 1)
            YB_l = pixcoordsloc[0][0]
            #YBloc['l'][i] = YB_l
            YBloc.loc[i,'l'] = YB_l
            YB_b = pixcoordsloc[0][1]
            #YBloc['b'][i] = YB_b
            YBloc.loc[i,'b'] = YB_b
            YB_radloc = YBloc['r'][i]
            YB_r = YB_radloc / pixscale8
            #YBloc['r'][i] = YB_r
            YBloc.loc[i, 'r'] = YB_r
            
        #Create cropped 100 x 100 pixel image arrays centered on YB
        #orig = image.um8data[y1:y2,x1:x2]
        #orig12 = image.um12data[y1:y2,x1:x2]
        #orig24 = image.um24data[y1:y2,x1:x2]
        
        #use Cutout2D to make the zoomed windows
        position = (YB_long_pix + 0.5, YB_lat_pix + 0.5)
        size = (100, 100)
        
        cut8 = Cutout2D(data=Mosaics.um8data,
                        position=position,
                        size=size,
                        wcs=Mosaics.um8w,
                        copy = True)
        '''cut12 = Cutout2D(data=image.um12data,
                         position=position,
                         size=size,
                         wcs=image.um12w)'''
        cut24 = Cutout2D(data=Mosaics.um24data,
                         position=position,
                         size=size,
                         wcs=Mosaics.um24w,
                         copy = True)
        '''cut70 = Cutout2D(data=image.um70data,
                         position=position,
                         size=size,
                         wcs=image.um70w)'''

        fitcopy8 = Mosaics.um8#.copy()
        fitcopy8.data = cut8.data
        fitcopy8.header.update(cut8.wcs.to_header())
        self.w = fitcopy8
        
        '''fitcopy12 = image.um12
        fitcopy12.data = cut12.data
        fitcopy12.header.update(cut12.wcs.to_header())'''

        fitcopy24 = Mosaics.um24#.copy()
        fitcopy24.data = cut24.data
        fitcopy24.header.update(cut24.wcs.to_header())

        '''fitcopy70 = image.um70
        fitcopy70.data = cut70.data
        fitcopy70.header.update(cut70.wcs.to_header())'''
        
        orig = cut8.data
        orig24 = cut24.data
        
        #create copies of cropped images called workmasks
        workmask8 = copy.deepcopy(orig)
        workmask24 = copy.deepcopy(orig24)
        
    

        
        self.skip8 = False
        self.skip24 = False
        
        img = workmask8
        header = 'vertices 8'
        if userdata[YBnum][header] != '':
            vertices = ast.literal_eval(userdata[YBnum][header])
            interp = do_interp(img, vertices)
            self.resid8 = interp.resid#.copy()
        else:
            print(f'No coordinates for YB {YBnum} at 8um')
            self.resid8 = np.zeros_like(workmask8)
            self.skip8 = True
                
        img = workmask24
        header = 'vertices 24'
        if userdata[YBnum][header] != '':
            vertices = ast.literal_eval(userdata[YBnum][header])
            interp = do_interp(img, vertices)
            self.resid24 = interp.resid#.copy()
        else:
            print(f'No coordinates for YB {YBnum} at 24um')
            self.resid24 = np.zeros_like(workmask8)
            self.skip24 = True
    
        
     




#########################################################################################################
'''
    Actual Program Starts here
    #'''
#########################################################################################################

#determines if the function should make the graphs
make_plots = True

#catalog_name = "DR2-YB-SMOG_wID.csv" #"DR2-YB-CygX_wID.csv" 
path = '.' #code says this is unused, it is in fact used. 
catalog_name = os.path.join(path, 'USE_THIS_CATALOG_ybcat_MWP_with_ID.csv')
#reads files
data = ascii.read(catalog_name, delimiter = ',')

if not os.path.exists(output_directory):
    # Create the output directory folder if it doesn't exist
    os.makedirs(output_directory)
    
    
if not os.path.exists(outfilename):
    #creats outfile if it doesnt exist
    column_names = 'YB ID, l, b, User Radius, 8um Amplitude (MJy/Sr), D[8um Amplitude], 8um x mean (pix), D[8um x mean ], 8um y mean (pix), D[8um y mean], 8um dist from center (pix), 8um x sd dev (arcsec), D[8um x sd dev ], 8um y st dev (arcsec), D[8um y st dev], 8 um theta, D[8 um theta], 8um residual (Jy), 8um residual/input flux (ratio), 24um amplitude (MJy/Sr), D[24um amplitude, 24um x mean (pix), D[24um x mean], 24um y mean (pix), D[24um y mean], 24um Dist from center (pix), 24um x sd dev (arcsec), D[24um x sd dev], 24um y st dev (arcsec), D[24um y st dev ], 24um theta, D[24um theta], 24um residual (Jy), 24um residual/input flux ratio, dist between means (arcsec), Patch Flag, classification string, cut value'
    df = pd.DataFrame(columns=column_names.split(', '))
    df.to_csv(outfilename, index=False)

'''YB1=data['ID'][0]
YB2=YB1+len(data) '''
cuts_count = np.full(len(data), -1)

temp_path70 = "temp_mosaic70"
temp_path24 = "temp_mosaic24"
temp_path12 = "temp_mosaic12"
temp_path8  = "temp_mosaic8"

startYB = 1
endYB = 6176
Mosaics = choose_image()

for i in range(startYB, endYB+1):  

    df = pd.read_csv(outfilename, index_col='YB ID')

    twocolor_image30 = AstroImage(i, Mosaics)
    if not twocolor_image30.skip:
        print("YB", i)
        twocolor_image30.show_gaussian(i)
    df.sort_values(by='YB ID', inplace=True)
    df.to_csv(outfilename, index=True)
    
os.remove(temp_path70)
os.remove(temp_path24)
os.remove(temp_path12)
os.remove(temp_path8)
#f.close()

histdata = ascii.read(output_directory + "2D_Gaussian_Fits.csv", delimiter = ',')
cut_values = histdata["cut value"]
cut_values = cut_values[~np.isnan(cut_values)]
plt.hist(cut_values, range = [-0.5, 4.5], bins = 5, ec = 'black') #make sure you make it big enough to get all the cut categories, if any more get added
plt.savefig(output_directory  + 'histogram.png')
print('\a')



