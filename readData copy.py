# from astropy.io import fits
# from astropy.table import Table
# import numpy as np
# import matplotlib.pylab as plt
# import matplotlib.lines as mlines
# from matplotlib.legend import Legend
# from pythonds.basic.stack import Stack
# from math import *
# from sklearn.neighbors import KDTree
# import healpy as hp
# from lrg_plot_functions import *
# from lrg_sum_functions import *
# from cosmo_Calc import *
# from divideByTwo import *
# # from readData import *
# from nearNeighbors import *
# from localBKG import *


# hdulist = fits.open('/Users/mtownsend/anaconda/Data/survey-dr7-specObj-dr14.fits') # this matches SDSS LRGs to DECaLS;
#                                                                  # ONLY GIVES SOURCES THAT ARE IN SDSS AND DECALS
# hdulist2 = fits.open('/Users/mtownsend/anaconda/Data/specObj-dr14.fits') # this is SDSS redshifts etc for LRGs
# hdulist3 = fits.open('/Users/mtownsend/anaconda/Data/sweep-240p005-250p010-dr7.fits') # this is one sweep file of the DECaLS data
# SpecObj_data = hdulist[1].data
# SDSS_data = hdulist2[1].data
# DECaLS_data = hdulist3[1].data


# A function to read in data from the Legacy Surveys and SDSS

def readData(SpecObj_data, SDSS_data, DECaLS_data):
    from astropy.io import fits
    from astropy.table import Table
    import numpy as np
    import matplotlib.pylab as plt
    import matplotlib.lines as mlines
    from matplotlib.legend import Legend
    from pythonds.basic.stack import Stack
    from sklearn.neighbors import KDTree
    import healpy as hp

    # Put data in arrays

    # Read in data from SDSS file

    # SDSS IDs useful for searching SDSS Science Archive Server

    plate = []
    plate = SDSS_data.field('PLATE')

    tile = []
    tile = SDSS_data.field('TILE')
    #
    # mjd = []
    # mjd = SDSS_data.file('MJD')

    # Redshift of galaxies according to sdss
    z = []
    z = SDSS_data.field('Z')

    # Unique ID for sources in SDSS
    specobjid = []
    specobjid = SDSS_data.field('SPECOBJID')

    # Class of object
    gal_class = []
    gal_class = SDSS_data.field('CLASS')

    # What survey the data is from
    survey = []
    survey = SDSS_data.field('SURVEY')

    # SPECPRIMARY; set to 1 for primary observation of object, 0 otherwise
    spec = []
    spec = SDSS_data.field('SPECPRIMARY')

    # Bitmask of spectroscopic warning values; need set to 0
    zwarn_noqso = []
    zwarn_noqso = SDSS_data.field('ZWARNING_NOQSO')

    # Spectroscopic classification for certain redshift?
    class_noqso = []
    class_noqso = SDSS_data.field('CLASS_NOQSO')

    # Array for LRG targets
    targets = []
    targets = SDSS_data.field('BOSS_TARGET1')

    # Section of code to find LRG targets

    def divideBy2(decNumber):

        # from pythonds.basic.stack import Stack
        # import numpy as np

        np.vectorize(decNumber)
        remstack = Stack()

        if decNumber == 0: return "0"

        while decNumber > 0:
            rem = decNumber % 2
            remstack.push(rem)
            decNumber = decNumber // 2

        binString = ""
        while not remstack.isEmpty():
            binString = binString + str(remstack.pop())

        return binString

    # Function to find LOWZ targets
    divideBy2Vec = np.vectorize(divideBy2)

    a = divideBy2Vec(targets)  # gives binary in string form

    b = []
    c = []

    for i in range(len(a)):
        b.append(list((a[i])))
        b[i].reverse()

    # print(b)

    lrg = []

    # Finds flags for BOSS LOWZ and CMASS sample
    for i in range(len(b)):
        try:
            if (b[i][0] == '1') or (b[i][1] == '1'):
                lrg.append(int(1))
            else:
                lrg.append(int(0))
        except IndexError:
            pass
            lrg.append(int(0))

    lrg = np.array(lrg)
    print('length of sdss array: ', len(lrg))
    print('length of lrg only array:', len(lrg[np.where(lrg == 1)]))

    # # Recommended SDSS cuts (from data paper) and cut to get only LRGs
    # SDSS_cuts = ((gal_class == 'GALAXY') & (spec == 1) & (zwarn_noqso == 0) & (class_noqso == 'GALAXY') & ((survey == 'sdss') | (survey == 'boss')) & (lrg == 1))
    #
    # z_LRG = z[np.where(SDSS_cuts)]
    # print('len z_LRG:' , len(z_LRG))

    # ------------------------------------------------------------------------------------------------------------

    # Read in data from SDSS row matched DECaLS file

    # Object ID from survey file; value -1 for non-matches
    objid_MATCHED = []
    objid_MATCHED = SpecObj_data.field('OBJID')
    # print('len objid_MATCHED:', len(objid_MATCHED))
    # print('len objid_MATCHED (non-matches): ', len(objid_MATCHED[np.where(objid_MATCHED == -1)]))
    # print('len objid_MATCHED (matches only): ', len(objid_MATCHED[np.where(objid_MATCHED > -1)]))

    # Add brickid
    brickid_MATCHED = []
    brickid_MATCHED = SpecObj_data.field('BRICKID')

    # Add brickname
    brickname_MATCHED = []
    brickname_MATCHED = SpecObj_data.field('BRICKNAME')

    # Only galaxies included
    gal_type_MATCHED = []
    gal_type_MATCHED = SpecObj_data.field('TYPE')

    # RA
    ra_MATCHED = []
    ra_MATCHED = SpecObj_data.field('RA')

    # Dec
    dec_MATCHED = []
    dec_MATCHED = SpecObj_data.field('DEC')

    # flux_g
    gflux_MATCHED = []
    gflux_MATCHED = SpecObj_data.field('FLUX_G')

    # flux_r
    rflux_MATCHED = []
    rflux_MATCHED = SpecObj_data.field('FLUX_R')

    # flux_z
    zflux_MATCHED = []
    zflux_MATCHED = SpecObj_data.field('FLUX_Z')

    # flux from WISE channel 1
    w1flux_MATCHED = []
    w1flux_MATCHED = SpecObj_data.field('flux_w1')

    # flux from WISE channel 2
    w2flux_MATCHED = []
    w2flux_MATCHED = SpecObj_data.field('flux_w2')

    # flux from WISE channel 3
    w3flux_MATCHED = []
    w3flux_MATCHED = SpecObj_data.field('flux_w3')

    # flux from WISE channel 4
    w4flux_MATCHED = []
    w4flux_MATCHED = SpecObj_data.field('flux_w4')

    # nobs == number of images that contribute to the central pixel
    # nobs_g
    gobs_MATCHED = []
    gobs_MATCHED = SpecObj_data.field('NOBS_G')

    # nobs_r
    robs_MATCHED = []
    robs_MATCHED = SpecObj_data.field('NOBS_R')

    # nobs_z
    zobs_MATCHED = []
    zobs_MATCHED = SpecObj_data.field('NOBS_Z')

    # depth in g
    # depth_g_MATCHED = DECaLS_data.field('galdepth_g')
    #
    # # depth in r
    # depth_r_MATCHED = DECaLS_data.field('galdepth_r')
    #
    # # depth in z
    # depth_z_MATCHED = DECaLS_data.field('galdepth_z')

    # Create a unique identifier by combinding BRICKID and OBJID

    id_MATCHED = []

    for i in range(len(objid_MATCHED)):
        if (objid_MATCHED[i] == -1):
            id_MATCHED.append(-1)
        else:
            temp1 = str(brickid_MATCHED[i]) + str(objid_MATCHED[i])
            id_MATCHED.append(temp1)

    # print('length of row matched targets in SDSS and DECaLS (matches only): ', len(id_MATCHED[np.where(id_MATCHED > -1)]))
    id_MATCHED = np.array(id_MATCHED)
    # ------------------------------------------------------------------------------------------------------------

    # Read in data from DECaLS bricks

    # Object ID from survey file
    objid_ALL = []
    objid_ALL = DECaLS_data.field('OBJID')
    # print(len(objid_ALL))

    # Add brickid
    brickid_ALL = []
    brickid_ALL = DECaLS_data.field('BRICKID')

    # Add brickname
    brickname_ALL = []
    brickname_ALL = DECaLS_data.field('BRICKNAME')

    # Only galaxies included
    gal_type_ALL = []
    gal_type_ALL = DECaLS_data.field('TYPE')

    # RA
    ra_ALL = []
    ra_ALL = DECaLS_data.field('RA')

    # Dec
    dec_ALL = []
    dec_ALL = DECaLS_data.field('DEC')

    # flux_g
    gflux_ALL = []
    gflux_ALL = DECaLS_data.field('FLUX_G')

    # flux_r
    rflux_ALL = []
    rflux_ALL = DECaLS_data.field('FLUX_R')

    # flux_z
    zflux_ALL = []
    zflux_ALL = DECaLS_data.field('FLUX_Z')

    # flux from WISE channel 1
    w1flux_ALL = []
    w1flux_ALL = DECaLS_data.field('flux_w1')

    # flux from WISE channel 2
    w2flux_ALL = []
    w2flux_ALL = DECaLS_data.field('flux_w2')

    # flux from WISE channel 3
    w3flux_ALL = []
    w3flux_ALL = DECaLS_data.field('flux_w3')

    # flux from WISE channel 4
    w4flux_ALL = []
    w4flux_ALL = DECaLS_data.field('flux_w4')

    # nobs == number of images that contribute to the central pixel
    # nobs_g
    gobs_ALL = []
    gobs_ALL = DECaLS_data.field('NOBS_G')

    # nobs_r
    robs_ALL = []
    robs_ALL = DECaLS_data.field('NOBS_R')

    # nobs_z
    zobs_ALL = []
    zobs_ALL = DECaLS_data.field('NOBS_Z')

    # depth in g
    depth_g_ALL = DECaLS_data.field('galdepth_g')

    # depth in r
    depth_r_ALL = DECaLS_data.field('galdepth_r')

    # depth in z
    depth_z_ALL = DECaLS_data.field('galdepth_z')

    # inverse variance of the flux in g, r, and z
    flux_ivar_g_ALL = DECaLS_data.field('flux_ivar_g')
    flux_ivar_r_ALL = DECaLS_data.field('flux_ivar_r')
    flux_ivar_z_ALL = DECaLS_data.field('flux_ivar_z')

    id_ALL = []

    for i in range(len(objid_ALL)):
        temp2 = str(brickid_ALL[i]) + str(objid_ALL[i])
        id_ALL.append(temp2)

    print('length of DECaLS targets in brick: ', len(id_ALL))

    id_ALL = np.array(id_ALL)

    print('length of id_ALL: ', len(id_ALL))

    # ------------------------------------------------------------------------------------------------------------

    # Make cuts to separate LRGs and background galaxies

    # Selects only LRGs (with other cuts
    # LRG_cut = ((gobs_MATCHED >= 2.) & (robs_MATCHED >= 2.) & (zobs_MATCHED >= 2.) & (gflux_MATCHED > 0.) & (rflux_MATCHED > 0.) & (zflux_MATCHED > 0.) & (w1flux_MATCHED > 0.) & (w2flux_MATCHED > 0.) & (w3flux_MATCHED > 0.) & (w4flux_MATCHED > 0.) & (objid_MATCHED > -1) & (lrg == 1) & ((gal_type_MATCHED == 'SIMP') | (gal_type_MATCHED == "DEV") | (gal_type_MATCHED == "EXP") | (gal_type_MATCHED == "REX")) & (ra_MATCHED >= 241) & (ra_MATCHED <= 246) & (dec_MATCHED >= 6.5) & (dec_MATCHED <= 11.5) & (gal_class == 'GALAXY') & (spec == 1) & (zwarn_noqso == 0) & (class_noqso == 'GALAXY') & ((survey == 'sdss') | (survey == 'boss')))
    LRG_cut = ((gobs_MATCHED >= 2.) & (robs_MATCHED >= 2.) & (zobs_MATCHED >= 2.) & (objid_MATCHED > -1) & (lrg == 1) & ((gal_type_MATCHED == 'SIMP') | (gal_type_MATCHED == "DEV") | (gal_type_MATCHED == "EXP") | (gal_type_MATCHED == "REX")) & (ra_MATCHED >= 241) & (ra_MATCHED <= 246) & (dec_MATCHED >= 6.5) & (dec_MATCHED <= 11.5) & (gal_class == 'GALAXY') & (spec == 1) & (zwarn_noqso == 0) & (class_noqso == 'GALAXY') & ((survey == 'sdss') | (survey == 'boss')))

    print(type(LRG_cut))
    # id_LRG = []
    # print(type(id_LRG))
    id_LRG = id_MATCHED[np.where(LRG_cut)]
    print('length of id_MATCHED with LRG_cut (id_LRG):', len(id_LRG))

    idcut = []

    # This creates a list that is the length of id_ALL that matches LRGs from the DECaLS/SDSS file to the DECaLS file
    # Use id_cut_noLRG == 0 to get galaxy sources that are NOT identified LRGs
    # For use in narrowing down DECaLS-only file (ie 'ALL')
    for i in range(len(id_ALL)):
        if any(id_LRG == id_ALL[i]):
            idcut.append(1)
        else:
            idcut.append(0)

    idcut = np.array(idcut)
    print('length of idcut:', len(idcut))
    print('length of idcut = 1 (is an LRG in DECaLS-only file):', len(idcut[np.where(idcut == 1)]))
    print('length of idcut = 0 (is not an LRG in DECaLS-only file):', len(idcut[np.where(idcut == 0)]))

    z_lrg = []
    plate_lrg = []
    tile_lrg = []
    specobjid_lrg = []
    # mjd_lrg = []
    ra_lrg = []
    dec_lrg = []
    objid_lrg = []
    brickid_lrg = []
    for i in range(len(id_ALL)):
        if (idcut[i] == 1):
            z_lrg.append(z[np.where(id_MATCHED == id_ALL[i])])
            ra_lrg.append(ra_MATCHED[np.where(id_MATCHED == id_ALL[i])])
            dec_lrg.append(dec_MATCHED[np.where(id_MATCHED == id_ALL[i])])
            plate_lrg.append(plate[np.where(id_MATCHED == id_ALL[i])])
            tile_lrg.append(tile[np.where(id_MATCHED == id_ALL[i])])
            specobjid_lrg.append(specobjid[np.where(id_MATCHED == id_ALL[i])])
            # mjd_lrg.append(mjd[np.where(id_MATCHED == id_ALL[i])])
            objid_lrg.append(objid_MATCHED[np.where(id_MATCHED == id_ALL[i])])
            brickid_lrg.append(brickid_MATCHED[np.where(id_MATCHED == id_ALL[i])])

    print('length of z_lrg:', len(z_lrg))
    z_lrg = np.array(z_lrg)
    z_LRG = np.concatenate(z_lrg)
    print('length of z_LRG:', len(z_LRG))
    ra_lrg = np.array(ra_lrg)
    ra_LRG = np.concatenate(ra_lrg)
    dec_lrg = np.array(dec_lrg)
    dec_LRG = np.concatenate(dec_lrg)
    plate_lrg = np.array(plate_lrg)
    plate_LRG = np.concatenate(plate_lrg)
    tile_lrg = np.array(tile_lrg)
    tile_LRG = np.concatenate(tile_lrg)
    specobjid_lrg = np.array(specobjid_lrg)
    specobjid_LRG = np.concatenate(specobjid_lrg)
    objid_lrg = np.array(objid_lrg)
    objid_LRG = np.concatenate(objid_lrg)
    brickid_lrg = np.array(brickid_lrg)
    brickid_LRG = np.concatenate(brickid_lrg)
    # mjd_lrg = np.array(mjd_lrg)
    # mjd_LRG = np.concatenate(mjd_lrg)

    # LRG_cut = ((id_cut_LRG == 1) & (gobs_MATCHED >= 3.) & (robs_MATCHED >= 3.) & (gflux_MATCHED > 0.) & (rflux_MATCHED > 0.) & (objid_MATCHED > -1) & (lrg == 1) & ((gal_type_MATCHED == 'SIMP') | (gal_type_MATCHED == "DEV") | (gal_type_MATCHED == "EXP") | (gal_type_MATCHED == "REX")) & (ra_MATCHED >= 241) & (ra_MATCHED <= 246) & (dec_MATCHED >= 6.5) & (dec_MATCHED <= 11.5) & (gal_class == 'GALAXY') & (spec == 1 ) & (zwarn_noqso == 0) & (class_noqso == 'GALAXY') & ((survey == 'sdss') | (survey == 'boss')))
    # & (brickid_LRG == brickid_ALL)
    # print(len(LOWZ_cut))

    # Cut out LRGs
    no_LRG_cut = ((idcut == 0) & (gobs_ALL >= 2.) & (robs_ALL >= 2.) & (zobs_ALL >= 2.) & (gflux_ALL > 0.) & (rflux_ALL > 0.) & (zflux_ALL > 0.) & ((gal_type_ALL == 'SIMP') | (gal_type_ALL == "DEV") | (gal_type_ALL == "EXP") | (gal_type_ALL == "REX")) & (ra_ALL >= 241) & (ra_ALL <= 246) & (dec_ALL >= 6.5) & (dec_ALL <= 11.5))
    # no_LRG_cut = ((idcut == 0) & (gobs_ALL >= 2.) & (robs_ALL >= 2.) & (zobs_ALL >= 2.) & (gflux_ALL > 0.) & (rflux_ALL > 0.) & (zflux_ALL > 0.) & (w1flux_ALL > 0.) & (w2flux_ALL > 0.) & (w3flux_ALL > 0.) & (w4flux_ALL > 0.) & ((gal_type_ALL == 'SIMP') | (gal_type_ALL == "DEV") | (gal_type_ALL == "EXP") | (gal_type_ALL == "REX")) & (ra_ALL >= 241) & (ra_ALL <= 246) & (dec_ALL >= 6.5) & (dec_ALL <= 11.5))
    # no_LRG_cut = ((idcut == 0) & (gobs_ALL >= 2.) & (robs_ALL >= 2.) & (zobs_ALL >= 2.) & ((gal_type_ALL == 'SIMP') | (gal_type_ALL == "DEV") | (gal_type_ALL == "EXP") | (gal_type_ALL == "REX")) & (ra_ALL >= 241) & (ra_ALL <= 246) & (dec_ALL >= 6.5) & (dec_ALL <= 11.5))

    # Flux cuts

    # Flux in g for only LRGs
    gflux_LRG = gflux_ALL[np.where(idcut == 1)]

    # Flux in r for only LRGs
    rflux_LRG = rflux_ALL[np.where(idcut == 1)]

    # Flux in g for only LRGs
    zflux_LRG = zflux_ALL[np.where(idcut == 1)]

    # flux in W1 for only LRGs
    w1flux_LRG = w1flux_ALL[np.where(idcut == 1)]

    # flux in W2 for only LRGs
    w2flux_LRG = w2flux_ALL[np.where(idcut == 1)]

    # flux in W3 for only LRGs
    w3flux_LRG = w3flux_ALL[np.where(idcut == 1)]

    # flux in W4 for only LRGs
    w4flux_LRG = w4flux_ALL[np.where(idcut == 1)]

    # Flux in g for non-LRGs
    gflux_BKG = gflux_ALL[np.where(no_LRG_cut)]

    # Flux in r for non-LRGs
    rflux_BKG = rflux_ALL[np.where(no_LRG_cut)]

    # Flux in z for non-LRGs
    zflux_BKG = zflux_ALL[np.where(no_LRG_cut)]

    # flux in W1 for non-LRGs
    w1flux_BKG = w1flux_ALL[np.where(no_LRG_cut)]

    # flux in W2 for non-LRGs
    w2flux_BKG = w2flux_ALL[np.where(no_LRG_cut)]

    # flux in W3 for  non-LRGs
    w3flux_BKG = w3flux_ALL[np.where(no_LRG_cut)]

    # flux in W4 for non-LRGs
    w4flux_BKG = w4flux_ALL[np.where(no_LRG_cut)]

    # Obs cuts

    # Number of images in g for only LRGs
    gobs_LRG = gobs_ALL[np.where(idcut == 1)]

    # Number of images in r for only LRGs
    robs_LRG = robs_ALL[np.where(idcut == 1)]

    # Number of images in g for only LRGs
    zobs_LRG = zobs_ALL[np.where(idcut == 1)]

    # Number of images in g for all galaxies in DECaLS
    gobs_BKG = gobs_ALL[np.where(no_LRG_cut)]

    # Number of images in r for all galaxies in DECaLS
    robs_BKG = robs_ALL[np.where(no_LRG_cut)]

    # Number of images in z for all galaxies in DECaLS
    zobs_BKG = zobs_ALL[np.where(no_LRG_cut)]

    # gmag_LRG = 22.5 - 2.5 * np.log10(gflux_LRG)
    # rmag_LRG = 22.5 - 2.5 * np.log10(rflux_LRG)
    # zmag_LRG = 22.5 - 2.5 * np.log10(zflux_LRG)
    #
    # color_LRG = gmag_LRG - rmag_LRG
    # # color_LRG = rmag_LRG - zmag_LRG
    # # gzcolor_LRG = gmag_LRG - zmag_LRG
    #
    # gmag_BKG = 22.5 - 2.5 * np.log10(gflux_BKG)
    # rmag_BKG = 22.5 - 2.5 * np.log10(rflux_BKG)
    # zmag_BKG = 22.5 - 2.5 * np.log10(zflux_BKG)
    #
    # color_BKG = gmag_BKG - rmag_BKG
    # # color_BKG = rmag_BKG - zmag_BKG
    # # gzcolor_BKG = gmag_BKG - zmag_BKG

    # depth cuts

    # depth in g for only LRGs
    gdepth_LRG = depth_g_ALL[np.where(idcut == 1)]

    # depth in r for only LRGs
    rdepth_LRG = depth_r_ALL[np.where(idcut == 1)]

    # depth in z for only LRGs
    zdepth_LRG = depth_z_ALL[np.where(idcut == 1)]

    # depth in g for all galaxies in DECaLS
    gdepth_BKG = depth_g_ALL[np.where(no_LRG_cut)]

    # depth in r for all galaxies in DECaLS
    rdepth_BKG = depth_r_ALL[np.where(no_LRG_cut)]

    # depth in z for all galaxies in DECaLS
    zdepth_BKG = depth_z_ALL[np.where(no_LRG_cut)]

    # inverse variance cuts
    # inverse variance for only LRGs
    # flux_ivar_g_LRG = flux_ivar_g_ALL[np.where(idcut == 1)]
    # flux_ivar_r_LRG = flux_ivar_r_ALL[np.where(idcut == 1)]
    # flux_ivar_z_LRG = flux_ivar_z_ALL[np.where(idcut == 1)]

    # inverse variance for all galaxies in DECaLS

    # flux_ivar_g_BKG = flux_ivar_g_ALL[np.where(no_LRG_cut)]
    # flux_ivar_r_BKG = flux_ivar_r_ALL[np.where(no_LRG_cut)]
    # flux_ivar_z_BKG = flux_ivar_z_ALL[np.where(no_LRG_cut)]

    # plt.hist(gmag_BKG, bins=50, color='green', alpha=0.5)
    # plt.hist(rmag_BKG, bins=50, color='red', alpha=0.5)
    # plt.hist(zmag_BKG, bins=50, color='lightblue', alpha=0.5)
    # plt.show()
    #
    # plt.hist(z_LRG, bins=50)
    # plt.show()

    ra_BKG = ra_ALL[np.where(no_LRG_cut)]
    dec_BKG = dec_ALL[np.where(no_LRG_cut)]

    # print("end readData")

    return id_ALL, ra_LRG, dec_LRG, ra_BKG, dec_BKG, z_LRG, gdepth_LRG, rdepth_LRG, zdepth_LRG, gdepth_BKG, rdepth_BKG, zdepth_BKG, gobs_LRG, robs_LRG, zobs_LRG, gobs_BKG, robs_BKG, zobs_BKG, gflux_LRG, rflux_LRG, zflux_LRG, gflux_BKG, rflux_BKG, zflux_BKG, w1flux_LRG, w2flux_LRG, w3flux_LRG, w4flux_LRG, w1flux_BKG, w2flux_BKG, w3flux_BKG, w4flux_BKG, plate_LRG, tile_LRG, specobjid_LRG, objid_LRG, brickid_LRG
    # return id_ALL, ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag_BKG, gmag_BKG, zmag_BKG, color_BKG, rmag_LRG, gmag_LRG, zmag_LRG, color_LRG, z_LRG, gdepth_LRG, rdepth_LRG, zdepth_LRG, gdepth_BKG, rdepth_BKG, zdepth_BKG, gobs_LRG, robs_LRG, zobs_LRG, gobs_BKG, robs_BKG, zobs_BKG, gflux_LRG, rflux_LRG, zflux_LRG, gflux_BKG, rflux_BKG, zflux_BKG
    # return id_ALL, ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag_BKG, gmag_BKG, zmag_BKG, grcolor_BKG, rzcolor_BKG, gzcolor_BKG, rmag_LRG, gmag_LRG, zmag_LRG, grcolor_LRG, rzcolor_LRG, gzcolor_LRG, z_LRG, gdepth_LRG, rdepth_LRG, zdepth_LRG, gdepth_BKG, rdepth_BKG, zdepth_BKG, gobs_LRG, robs_LRG, zobs_LRG, gobs_BKG, robs_BKG, zobs_BKG, gflux_LRG, rflux_LRG, zflux_LRG, gflux_BKG, rflux_BKG, zflux_BKG
    # return id_ALL, ra_LRG, dec_LRG, ra_BKG, dec_BKG, z_LRG, gdepth_LRG, rdepth_LRG, zdepth_LRG, gdepth_BKG, rdepth_BKG, zdepth_BKG, gobs_LRG, robs_LRG, zobs_LRG, gobs_BKG, robs_BKG, zobs_BKG, flux_ivar_g_LRG, flux_ivar_r_LRG, flux_ivar_z_LRG, flux_ivar_g_BKG, flux_ivar_r_BKG, flux_ivar_z_BKG, gflux_LRG, rflux_LRG, zflux_LRG, gflux_BKG, rflux_BKG, zflux_BKG

# id_ALL, ra_LRG, dec_LRG, ra_BKG, dec_BKG, rmag_BKG, gmag_BKG, zmag_BKG, color_BKG, rmag_LRG, gmag_LRG, zmag_LRG, color_LRG, z_LRG, gdepth_LRG, rdepth_LRG, zdepth_LRG, gdepth_BKG, rdepth_BKG, zdepth_BKG = readData(SpecObj_data, SDSS_data, DECaLS_data)

# plt.scatter(ra_BKG, dec_BKG, s=1, color='blue')
# plt.scatter(ra_LRG, dec_LRG, s=1, color='red')
# plt.rcParams["figure.figsize"] = [15, 15]
# plt.show()
#
# row = 10
# column = 10
# # creates histogram for survey sources; excludes LRGs
# H, xedges, yedges = np.histogram2d(rmag_BKG, color_BKG, normed=False)
#
# cmd(rmag_BKG, color_BKG, rmag_LRG, color_LRG, xedges, yedges)
# plt.show()

# print('end readData')
