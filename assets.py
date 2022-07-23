# Check we can load astroquery packages, that we will use later.
import sys
import cv2
from astropy.io import fits
from astroquery.mast import Observations


def get_obs(dataset, filter):
    d_that_matches_filter = None
    d_that_maybe_matches_obs_id = None
    for d in dataset:
        print(f'filter is {d["filters"]}')
        # print(f'wavelength is {d["wavelength_region"]}')
        for x in d.colnames:
            # print(f'{x}::{d[x]}')
            if d['filters'] == filter:
                d_that_matches_filter= d
            if filter.lower() in d['obs_id'].lower():
                d_that_maybe_matches_obs_id = d
    if d_that_matches_filter is None and d_that_maybe_matches_obs_id is not None:
        returned_d_that_probably_matches_filter = d_that_maybe_matches_obs_id #handles situation like in carina nebula F470N isn't in the filter name but is in the obs_id name.
    elif d_that_matches_filter is not None:
        returned_d_that_probably_matches_filter = d_that_matches_filter
    else:
        returned_d_that_probably_matches_filter = None
    return returned_d_that_probably_matches_filter

def make_image(data, outputname, size=(1, 1), dpi=2000, cmap='gray'):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap(cmap)
    ax.imshow(data, interpolation='nearest', vmin=0, vmax=50)
    plt.savefig(outputname, dpi=dpi)

#KEY
#SMACS = gravitational lensing
#NGC 3324 = cosmic cliffs near carina nebula
#NGC 3132 = ring nebula
#NGC 7320 = stephan's quintet

obj=["SMACS J0723.3-7327", "NGC 3324", "NGC 3324", "NGC 3132", "NGC 7320"]
NGC_3132_filters=["F090W", "F200W", "F444W", "F770W", "F1130W", "F1280W", "F1800W"]
NGC_3324_NIRCAM_filters = ["F200W","F187N","F090W","F335M","F444W"] #"F470N" is really small so it throws off the size
NGC_3324_MIRI_filters = ["F770W","F1130W","F1280W","F1800W"]
filters = [[],NGC_3324_NIRCAM_filters, NGC_3324_MIRI_filters, NGC_3132_filters, []]
obj_num=1
object = obj[obj_num]
single_filter = filters[obj_num][4]
dpi = 1  # interpolated. need to do this automatically
wid = None
ht  = None
outimgs = []
show_plots=False

'''Test that we can talk to MAST via Astroquery below.'''

obsByName = Observations.query_object(object,radius=".2 deg")
print("Number of results from all missions:",len(obsByName))
print(obsByName[:10])

if len(obsByName) <=0:
    #nothing found
    print("NO RESULTS!")
    sys.exit()

# See if you can see any observations by JWST!
jwst_only = obsByName[(obsByName['obs_collection'] == 'JWST')]
print("\n\n JWST results only \n")
print(jwst_only)


'''Choosing an Observation
Let's concentrate on the NIRCAM observations, just because I think the image is prettier. Each observation is made up of "data products", which we can list and have a look at in a moment.

Let's start by looking at the available fields in our dataset, using the .keys() method, in order to choose which observation to use. Of interest is that there's an obs_id and an obsid, one of which is a more human readable id than the other, but both function as unique ids.'''
print(jwst_only.keys(), '\n\n')
print(jwst_only['obsid', 'instrument_name', 'filters', 'obs_id', 't_min'])

'''The NIRCAM obserations seem to be mainly differentiated by filters used. There are 29 different NIRCAM filters each giving you images of different parts of the wavelength spectrum. In order to make a beautiful composite image like the one released, you'd want to combine the images from the different filters, but for the simplicity of this tutorial let's just pick one with F444W as that seems to cover a broad range of wavelengths in the diagram.

I'll arbitrarily pick the one that started first (smaller t_min): jw02733-o001_t001_nircam_f405n-f444w / 87602459.'''

if show_plots:
    filter_list = [single_filter]
else:
    filter_list = filters[obj_num]

for obs_filter in filter_list:
    # Pick a particular observation from the list.
    obs_87602459 = jwst_only[(jwst_only['obsid'] == '87602459')]
    obs= get_obs(jwst_only, filter=obs_filter)

    if obs is None:
        print(f"Filter {obs_filter} doesn't exist for this object")
        sys.exit()

    print(f"\n Observation {obs} of {object}: \n")
    print(obs['obsid'])
    print(obs['filters'])

    # Get a list of the "Data Products" available from the observations.
    dataProducts = Observations.get_product_list(obs['obsid'])
    print("\n Data Products: \n")
    print(dataProducts)

    '''Choosing a Data Product
    As you can see there are lots of data products. In order to narrow them down, lets look at the calib_level value, which tells us how "raw" the data is. This is important, because data coming directly from the telescope is processed and calibrated to increasining degrees of refinement, so we generally want the highest possible calibration level.
    
    Level 3 is defined as "Combined, calibrated science product per target or source", which sounds pretty good. Also, as you can see below, there's a much more manageable number of calib_level 3 data products to contend with:'''

    calibrated = dataProducts[(dataProducts['calib_level'] >= 3)]
    print("Found ", len(calibrated), " data products with calibration level >= 3: \n")
    print(calibrated)

    # See how we can differentiate the products
    print("\n",calibrated.keys())
    print(calibrated['obsID','productSubGroupDescription','description'])


    '''Which of these should we download? Reading through the docs on the data product formats, the I2D sounds most like what we want- image data we can display. Let's download it and see.
    '''
    i2d = calibrated[(calibrated["productSubGroupDescription"] == "I2D")]

    # Note that mrp_only should be redundant here- it means "Minimum Recommended Products", which are
    # products calibrated finely enough to be reccomended for science use. As we already filtered to
    # calib_level >= 3, both our products should be downloaded.
    manifest = Observations.download_products(i2d,mrp_only=True)


    '''3. Loading and Viewing Downloaded Data
    Hooray, we successfully downloaded real JWST data! You can see the file extension is .fits, which is a common file format in astronomy. Its docs are here, and we imported a fits module from astropy earlier to deal with it.
    
    Lets load in our downloaded data and see if we can display it. I checked the astropy docs on reading images, which gave me the imports below, but you could probably manage with just standard matplotlib.'''

    import numpy as np
    import matplotlib.pyplot as plt
    # %matplotlib inline


    # Importing some special packages to display astronomy data.
    from matplotlib.colors import LogNorm
    from astropy.visualization import astropy_mpl_style
    plt.style.use(astropy_mpl_style)

    dp_87602459 = fits.open(manifest['Local Path'][0])

    # We can see there are 9 HDUs in the file of various types - one Primary HDU, 7
    # ImageHDUs, and 2 trailing BinTableHDUs.
    dp_87602459.info()


    '''Figuring out the shape of the data
    We can use the docs to understand shape of our data we saw in the output of .info() above. Of particular interest in the docs is:
    
    The file consists of an HDU list, made up of Header Data Units, which each contain data and a header object that describes the data.
    There is a leading "Primary" HDU containing information about the data product as a whole (but no data), and a trailing HDU containing "ADSF"- the data model metadata. We saw the primary HDU in the cell above.
    For our data product type (I2D), the "SCI" ext name seems to indicate something interesting we could use: "SCI: 2-D data array containing the pixel values, in units of surface brightness". We can see in the .info() output that the SCI data is stored at index 1.
    Next, lets look at the SCI object and its header.'''


    # Looking at the header for the second HDU at index 1, we see it describes
    # the data it contains, and that it matches the description in the docs - a
    # 2D array, with various pieces of metadata. NAXIS1 and NAXIS2 tell us the
    # number of pixels in each dimension of the array- so this is a big image!
    print("Header of HDU 1 - the 'SCI' ")
    dp_87602459[1].header

    '''Finally, a picture!
    Now that we've looked at the header for the 'SCI' object, and understand what the data's shape is and its meaning, lets look at the data itself.'''

    data = dp_87602459[1].data
    # As expected, the data is a 2D numpy array
    print('data shape: ', np.shape(data))
    wid_pixels = np.shape(data)[1]
    ht_pixels = np.shape(data)[0]
    if wid == None:
        wid = wid_pixels / 1 #for images around 14,500 pixels, that's a 14.5 inch file
        ht = ht_pixels / 1

        plt.rcParams['figure.figsize'] = [wid, ht]
        plt.rcParams['figure.dpi'] = dpi
        wid_min = wid_pixels
        ht_min = ht_pixels
        print(f"set fig size to {wid}x{ht}")
    else:
        if wid_pixels < wid_min:
            print(f"setting wid_min from {wid_min} to {wid_pixels}")
            wid_min = wid_pixels

        if ht_pixels < ht_min:
            print(f"setting wid_min from {ht_min} to {ht_pixels}")
            ht_min = ht_pixels

    # # Let's plot it as an image and see what we get:
    # plt.figure()
    # plotmg = plt.imshow(data, cmap='gray')
    # plt.show()
    # plt.colorbar(plotmg)

    '''hmm, pretty underwhelming stuff. Let's troubleshoot. There is definitely data present, and looking at the historgram data below, you can see there's massive dynamic range in the image (lots of basically zero values, and a few very bright ones).'''
    print(np.sum(data))
    # The initial array shows us how many pixels fall into the corresponding "bin" in the same index
    # of the second array.
    np.histogram(data)

    if show_plots:

        plotmg = plt.imshow(data, cmap='CMRmap', interpolation='nearest',
                            vmin=0, vmax=50)
        plt.grid(b=None)
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.show()

        plotmg3 = plt.imshow(data, cmap='nipy_spectral',interpolation='nearest',
                       vmin=0, vmax=50)
        plt.grid(b=None)
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.show()

        plotmg2 = plt.imshow(data, cmap='afmhot',interpolation='nearest',
                       vmin=0, vmax=50)
        plt.grid(b=None)
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.show()

    else:
        outname=f"{object}_{obs_filter}.tiff".replace(" ","_")

        # fig = plt.imshow(data, cmap='gray', interpolation='nearest', vmin=0, vmax=50)
        # # plt.plot(data)
        # plt.grid(b=None)
        # plt.gca().invert_yaxis()
        # plt.axis('off')
        # # dpi = 2339  # interpolated. need to do this automatically
        #
        # horrible_wid = float(str(fig).split(",")[1].split(';')[1].split('x')[0])
        # horrible_dpi = dpi* wid/horrible_wid
        # fig.axes.get_xaxis().set_visible(False)
        # fig.axes.get_yaxis().set_visible(False)
        # plt.savefig(outname,  bbox_inches='tight', pad_inches=0, dpi=horrible_dpi)

        cv2.imwrite(outname, data)

        outimgs.append(outname)

# for img_file in outimgs:
#     image = cv2.imread(img_file)
#     outname_cropped = f"cropped_{img_file}"
#     y=0
#     x=0
#     crop_image = image[x:wid_min, y:ht_min]
#     cv2.imwrite(outname_cropped, image)

