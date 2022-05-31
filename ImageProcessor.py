import ray
import os
from datetime import datetime
from glob import glob
import json
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.morphology import disk
import skimage.segmentation


class ImageProcessor:
    '''This class will read images and store some basic information into a data dictionary.'''
    def __init__(self, in_folder, ext='tif'):
        '''Load images, compute basic image stats and store them 
        Parameters
        ----------
        in_folder : str
            Location of folder containing images to process
        ext : str
            Extension of images (default: *.tif)
        Attributes
        ----------
        data : dictionary
            This dictionary follows the below structure:
            data = 
            {
                "timestamp": datetime,
                "inputs": [
                {
                    "image" : 2d list of int,
                    "image_path": str,
                    "image_stat": {
                        "height" : int,
                        "width" : int,
                        "max_intensity": int,
                        "min_intensity": int,
                        "avg_intensity": float,
                        "std_intensity": float,
                        "hist_intensity": 2d list of float
                        }
                },
                ...
                ]
            }
        '''
        image_files = sorted(glob(os.path.join(in_folder,'*.'+ext)))
        print("Found {} image(s)".format(len(image_files)))
        
        self.data = {}
        self.data['timestamp'] = str(datetime.now())
        self.data['inputs'] = []
        for image_file in image_files:
            # Load image
            image = cv2.imread(image_file,0)
            if image is None:
                raise ValueError("Unable to load image..")
            
            # Compute image stats
            height,width = image.shape
            min_intensity = np.min(image)
            max_intensity = np.max(image)
            avg_intensity = np.mean(image)
            std_intensity = np.std(image)
            hist_intensity = cv2.calcHist([image], [0], None, [int(max_intensity)], [int(min_intensity),int(max_intensity)])

            # Store image info
            _image_stats = {}
            _image_stats['height'] = int(height)
            _image_stats['width']  = int(width)
            _image_stats['min_intensity']  = int(min_intensity)
            _image_stats['max_intensity']  = int(max_intensity)
            _image_stats['avg_intensity']  = float(avg_intensity)
            _image_stats['std_intensity']  = float(std_intensity)
            _image_stats['hist_intensity'] = hist_intensity.tolist()
            _data = {}
            _data["image"] = image.tolist()
            _data['image_path'] = image_file
            _data['image_stat'] = _image_stats
            self.data['inputs'].append(_data)
    
    def segmentByGraphCut(self, inp_01=None, inp_02=None, inp_03=None, method='felzenszwalb', scale=3000, kernel_size=(15,15)):
        '''Segment image based on graph cut algorithm
        Parameters
        ----------
        inp_01 : dict
            A dictionary object of the first channel of an image.
        inp_02 : dict
            A dictionary object of the second channel of an image.
        inp_03 : dict
            A dictionary object of the third channel of an image.
        method : str
            For now only 'felzenszwalb' option is available.
        scale : int
            Determines the degree of finer/coarser.
            Larger scale value loads to finer segmentation result.
        kernel : (int,int)
            Kernel size for morphological closing
            
        Returns
        -------
        image_pred : ndarray (uint8)
            Segmentation result (binary mask, 0:background 1:object)
        '''
        # Get image
        image_01 = np.array(inp_01['image']).astype(np.uint8)
        image_02 = np.array(inp_02['image']).astype(np.uint8)
        image_03 = np.array(inp_03['image']).astype(np.uint8)
        
        if method=='felzenszwalb':
            # Based on the observation that two channels seem to be more responsible to noise than the other one, 
            # compute a potential noise-like object using two channels and then perform logical operand to exclude them.
            # Specifically, image_01 and image_02 are for the noise, and image_03 is for the region of interest.
            pred_01 = skimage.segmentation.felzenszwalb(image_01, scale=scale)
            pred_01[pred_01>0] = 1
            pred_01 = pred_01.astype(bool)
            pred_02 = skimage.segmentation.felzenszwalb(image_02, scale=scale)
            pred_02[pred_02>0] = 1
            pred_02 = pred_02.astype(bool)
            pred_03 = skimage.segmentation.felzenszwalb(image_03, scale=scale)
            pred_03[pred_03>0] = 1
            pred_03 = pred_03.astype(bool)
            # Exclude noise-like instances from the prediction
            pred_potentialnoise = np.logical_and(pred_01,pred_02)
            pred_noisefree  = np.logical_and(np.logical_not(pred_potentialnoise),pred_03)
            # Post-process prediction using (1) filling out contours and (2) morphological closing
            # (1) Filling contours
            contours = cv2.findContours(pred_noisefree.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = contours[-2] 
            pred = np.zeros((inp_03['image_stat']['height'],inp_03['image_stat']['width']))
            cv2.fillPoly(pred,pts=contours,color=(255,255,255))
            # (2) Morphological closing
            pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, np.ones(kernel_size))
            pred = pred//255
        else:
            raise NotImplemented 
        return pred.astype(np.uint8)
    
    @ray.remote
    def segmentByGraphCutDist(i=0, image_01=None, image_02=None, image_03=None, method='felzenszwalb', scale=3000, kernel_size=(15,15)):
        '''Segment image based on graph cut algorithm in a distributed fashion
        Parameters
        ----------
        i : int
            The index of CPU.
        inp_01 : dict
            A dictionary object of the first channel of an image.
        inp_02 : dict
            A dictionary object of the second channel of an image.
        inp_03 : dict
            A dictionary object of the third channel of an image.
        method : str
            For now only 'felzenszwalb' option is available.
        scale : int
            Determines the degree of finer/coarser.
            Larger scale value loads to finer segmentation result.
        kernel : (int,int)
            Kernel size for morphological closing
            
        Returns
        -------
        image_pred : ndarray (uint8)
            Segmentation result (binary mask, 0:background 1:object)
        '''
        # Get image
        #image_01 = np.array(inp_01['image']).astype(np.uint8)
        #image_02 = np.array(inp_02['image']).astype(np.uint8)
        #image_03 = np.array(inp_03['image']).astype(np.uint8)
        
        #print(i, image_01.shape, image_02.shape, image_03.shape)
        
        # Patch Generation
        '''Patch generation
        TODO
        ----
        1. Compute the original image size
        2. Compute the window index based on i, which will be vary depending on the # of CPU
            2.1. Raster-order
        3. Generate patch based on the window index

        if i == 0:
            image_01 = image_01[:2500,:2800]
            image_02 = image_02[:2500,:2800]
            image_03 = image_03[:2500,:2800]
        elif i == 1:
            image_01 = image_01[:2500,2800*1:2800*2]
            image_02 = image_02[:2500,2800*1:2800*2]
            image_03 = image_03[:2500,2800*1:2800*2]
        elif i == 2:
            image_01 = image_01[:2500,2800*2:2800*3]
            image_02 = image_02[:2500,2800*2:2800*3]
            image_03 = image_03[:2500,2800*2:2800*3]
        elif i == 3:
            image_01 = image_01[:2500,2800*3:]
            image_02 = image_02[:2500,2800*3:]
            image_03 = image_03[:2500,2800*3:]
        elif i == 4:
            image_01 = image_01[2500*1:2500*2,:2800]
            image_02 = image_02[2500*1:2500*2,:2800]
            image_03 = image_03[2500*1:2500*2,:2800]
        elif i == 4:
            image_01 = image_01[2500*1:2500*2,2800*1:2800*2]
            image_02 = image_02[2500*1:2500*2,2800*1:2800*2]
            image_03 = image_03[2500*1:2500*2,2800*1:2800*2]
        elif i == 5:
            image_01 = image_01[2500*1:2500*2,2800*2:2800*3]
            image_02 = image_02[2500*1:2500*2,2800*2:2800*3]
            image_03 = image_03[2500*1:2500*2,2800*2:2800*3]
        else:
            image_01 = image_01[2500*1:2500*2,2800*3:]
            image_02 = image_02[2500*1:2500*2,2800*3:]
            image_03 = image_03[2500*1:2500*2,2800*3:]
        '''
        # For now, consider to use fixed patch-size (1024x1024)
        patch_size = 1024
        image_01 = image_01[patch_size*3:patch_size*4, patch_size*3:patch_size*4]
        image_02 = image_02[patch_size*3:patch_size*4, patch_size*3:patch_size*4]
        image_03 = image_03[patch_size*3:patch_size*4, patch_size*3:patch_size*4]
        
        if method=='felzenszwalb':
            # Based on the observation that two channels seem to be more responsible to noise than the other one, 
            # compute a potential noise-like object using two channels and then perform logical operand to exclude them.
            # Specifically, image_01 and image_02 are for the noise, and image_03 is for the region of interest.
            pred_01 = skimage.segmentation.felzenszwalb(image_01, scale=scale)
            pred_01[pred_01>0] = 1
            pred_01 = pred_01.astype(bool)
            pred_02 = skimage.segmentation.felzenszwalb(image_02, scale=scale)
            pred_02[pred_02>0] = 1
            pred_02 = pred_02.astype(bool)
            pred_03 = skimage.segmentation.felzenszwalb(image_03, scale=scale)
            pred_03[pred_03>0] = 1
            pred_03 = pred_03.astype(bool)
            # Exclude noise-like instances from the prediction
            pred_potentialnoise = np.logical_and(pred_01,pred_02)
            pred_noisefree  = np.logical_and(np.logical_not(pred_potentialnoise),pred_03)
            # Post-process prediction using (1) filling out contours and (2) morphological closing
            # (1) Filling contours
            contours = cv2.findContours(pred_noisefree.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = contours[-2] 
            pred = np.zeros((patch_size,patch_size))
            cv2.fillPoly(pred,pts=contours,color=(255,255,255))
            # (2) Morphological closing
            pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, np.ones(kernel_size))
            pred = pred//255
        else:
            raise NotImplemented 
        return pred.astype(np.uint8)

    def segmentByThresholding(self, inp_01=None, inp_02=None, inp_03=None, sigma01=11, sigma02=11, sigma03=11, method="otsu"):
        '''Segment image based on classic thresholding techinques
        Parameters
        ----------
        inp_01 : dict
            A dictionary object of the first channel of an image.
        inp_02 : dict
            A dictionary object of the second channel of an image.
        inp_03 : dict
            A dictionary object of the third channel of an image.
        sigma01 : int
            A blurring parameter for the first channel of an image.
        sigma02 : int
            A blurring parameter for the second channel of an image.
        sigma03 : int
            A blurring parameter for the third channel of an image.
        method : str
            Thresholding option either "otsu" or "adaptive"

        Returns
        -------
        image_pred : ndarray (uint8)
            Segmentation result (binary mask, 0:background 1:object)
        '''
        # Blur images for removing noise
        image_01_blur = cv2.GaussianBlur(inp_01['image'], (sigma01, sigma01), 0)
        image_02_blur = cv2.GaussianBlur(inp_02['image'], (sigma02, sigma02), 0)
        image_03_blur = cv2.GaussianBlur(inp_03['image'], (sigma03, sigma03), 0)
        # Thresholding
        if method=="otsu":
            _, pred_01 = cv2.threshold(image_01_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            pred_01 = np.invert(pred_01)
            pred_01[pred_01>0] = 1
            _, pred_02 = cv2.threshold(image_02_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            pred_02 = np.invert(pred_02)
            pred_02[pred_02>0] = 1
            _, pred_03 = cv2.threshold(image_03_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            pred_03 = np.invert(pred_03)
            pred_03[pred_03>0] = 1
        elif method=="adaptive":
            pred_01 = cv2.adaptiveThreshold(image_01_blur, inp_01['image_stat']['max_intensity'], cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
            pred_02 = cv2.adaptiveThreshold(image_02_blur, inp_02['image_stat']['max_intensity'], cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
            pred_03 = cv2.adaptiveThreshold(image_03_blur, inp_03['image_stat']['max_intensity'], cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        else:
            raise NotImplemented
        # Fusing results
        pred = pred_01 + pred_02 + pred_03
        pred[pred>0] = 1
        return pred

    def computeMetrics(self, gt, pred):
        '''Validate segmentation result
        Parameters
        ----------
        gt : ndarray
            A 2d array of int (0:background 1:object).
        pred : ndarray
            A 2d array of int (0:background 1:object).
        
        Returns
        -------
        res : dict
            A dictionary of metrics
        '''
        def _computeConfMat(gt, pred):
            tn = np.sum(np.logical_and(gt==0, pred==0))
            tp = np.sum(np.logical_and(gt==1, pred==1))
            fn = np.sum(np.logical_and(gt==1, pred==0))
            fp = np.sum(np.logical_and(gt==0, pred==1))
            return tn, tp, fn, fp

        tn, tp, fn, fp = _computeConfMat(gt, pred)
        accuracy  = (tp+tn)/(tp+tn+fp+fn)
        precision = tp/(tp+fp)
        recall    = tp/(tp+fn)
        f1        = 2*tp/(2*tp+fp+fn)
        iou       = tp/(tp+fp+fn)
        res = {
            'acc' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'f1' : f1,
            'iou' : iou
        }
        return res
    
    def computeInstances(self, pred):
        '''Return features of the segmented objects
        Parameters
        ----------
        pred : ndarray
            A 2d array of int (0:background 1:object).
        
        Returns
        -------
        res : dict
            {
                "mask" : 2d list of int,
                "contours" : [
                {
                    "momentum" : (float,float),           # image moments
                    "area" : float,                       # contour area
                    "perimeter" : float,                  # arc length
                    "convhull" : 2d list of (int,int),    # points of polygon
                    "boundingbox" : [int, int, int, int], # points of bb (x, y, w, h)
                    "mincircle" : [float, float, float]   # minimum enclosing circle (x, y, and radius)
                },
                ...
                ]
            }
        '''        
        contours,_ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        res = {'mask' : pred.tolist()}
        res = {'contours' : []}
        for contour in contours:
            _res = {}
            # Moments
            mom = cv2.moments(contour)
            cx = int(mom['m10']/mom['m00'])
            cy = int(mom['m01']/mom['m00'])
            _res["momentum"] = (cx,cy)
            # Area
            area = cv2.contourArea(contour)
            _res["area"] = area
            # Perimeter
            perimeter = cv2.arcLength(contour,True)
            _res["perimeter"] = perimeter
            # Convex Hull
            hull = cv2.convexHull(contour)
            _res["convhull"] = hull.tolist()
            # Bounding box
            x,y,w,h = cv2.boundingRect(contour)
            _res["boundingbox"] = (x,y,w,h)
            # Minimum enclosing circle
            (x,y),r = cv2.minEnclosingCircle(contour)
            _res["mincircle"] = (x,y,r)
            res['contours'].append(_res)
        return res
    

'''
Parse Arguments
'''
parser = argparse.ArgumentParser(description='Processing medical image')
parser.add_argument('-i', type=str, required=True,
                   help='a path to the folder containing images to process.')
parser.add_argument('-g', type=str, required=False,
                   help='a path to the groundtruth file.')
parser.add_argument('-m', type=str, required=True,
                    help='a mode to run.')
args = parser.parse_args()

INPUT_PATH = args.i
MODE       = args.m
GT_PATH    = args.g

if MODE == "TQ1":
    ip = ImageProcessor(INPUT_PATH)
    print("Getting image info...")
    for _data in ip.data['inputs']:
        print("\nImage Info:")
        print("\timage_path : {}".format(_data['image_path']))
        print("\theight : {}".format(_data['image_stat']['height']))
        print("\twidth : {}".format(_data['image_stat']['width']))
        print("\tmin_intensity : {}".format(_data['image_stat']['min_intensity']))
        print("\tmax_intensity : {}".format(_data['image_stat']['max_intensity']))
        print("\tavg_intensity : {}".format(_data['image_stat']['avg_intensity']))
        print("\tstd_intensity : {}".format(_data['image_stat']['std_intensity']))
    
elif MODE == "TQ2":
    gt = cv2.imread(GT_PATH,0)
    ip = ImageProcessor(INPUT_PATH)
    print("Running segmentation...")
    pred = ip.segmentByGraphCut(ip.data['inputs'][0],ip.data['inputs'][1],ip.data['inputs'][2])
    print("Computing metrics...")
    metrics = ip.computeMetrics(gt,pred)
    print(metrics)
    save_path = "./TQ2_out.png"
    plt.figure()
    plt.imshow(pred)
    plt.savefig(save_path)
    print("Segmented out is stored at {}".format(save_path))
              
elif MODE == "TQ3":
    ip = ImageProcessor(INPUT_PATH)
    print("Running segmentation...")
    pred = ip.segmentByGraphCut(ip.data['inputs'][0],ip.data['inputs'][1],ip.data['inputs'][2])
    print("Computing features of objects...")
    data = ip.computeInstances(pred)
    save_path = "./TQ3_out.json"
    with open(save_path, 'w') as f:
        json.dump(data, f)
    print("The mask and objects' features are stored at {}".format(save_path))
        
elif MODE == "TQ4":
    print("Extracting signal differences...")
    image_paths = sorted(glob(os.path.join(INPUT_PATH+'/*.tif')))
    thresholds = [10,25,50,100,150]
    lines = {}
    for i in range(1,len(image_paths)):
        image_prev = cv2.GaussianBlur(cv2.imread(image_paths[i-1], 0), (9, 9), 0)
        image_curr = cv2.GaussianBlur(cv2.imread(image_paths[i]  , 0), (9, 9), 0)
        for threshold in thresholds:
            delta = cv2.absdiff(image_curr,image_prev)
            delta[delta<threshold] = 0
            delta[delta>=threshold] = 1
            amount_change = np.sum(delta)
            if threshold in lines:
                lines[threshold].append(amount_change)
            else:
                lines[threshold] = [amount_change]
    save_path = "./TQ4_out.png"
    plt.figure()
    for threshold in thresholds:
        plt.plot(lines[threshold], label = str(threshold))
    plt.legend()
    plt.savefig(save_path)
    print("The plot is stored at {}".format(save_path))

elif MODE == "TQ5":
    ip = ImageProcessor(INPUT_PATH)
    print("Running segmentation using Ray...")
    
    # Starting Ray
    ray.init()        
    #results = ray.get([ip.segmentByGraphCutDist.remote(i,ip.data['inputs'][0],ip.data['inputs'][1],ip.data['inputs'][2]) for i in range(os.cpu_count())])
    image_01 = np.array(ip.data['inputs'][0]['image']).astype(np.uint8)
    image_02 = np.array(ip.data['inputs'][1]['image']).astype(np.uint8)
    image_03 = np.array(ip.data['inputs'][2]['image']).astype(np.uint8)
    results = ray.get([ip.segmentByGraphCutDist.remote(i,image_01,image_02,image_03) for i in range(6)])
    # Shutdown Ray
    ray.shutdown()
    for idx,result in enumerate(results):
        save_path = "./TQ5_" + str(idx) + "_out.png"
        plt.figure()
        plt.imshow(result)
        plt.savefig(save_path)
        print("Segmented out is stored at {}".format(save_path))
    
else:
    raise NotImplemented