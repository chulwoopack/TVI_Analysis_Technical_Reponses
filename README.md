## Requirements
In order to run the scripts, it is necessary to install the following library on your system:
* [Pandas](https://pandas.pydata.org/) (>=0.24.2)

## TQ1 - Image Processing
When you instantiate `ImageProcessor`, it will read images from `TQ1/` folder, extract the following information, and store them into a `data` variable as a dictionary:
```JSON
{
    "timestamp": "",              // datetime
    "inputs": [
     {
         "image" : "",            // 2d list of int
         "image_path": "",        // str
         "image_stat": {
             "height" : "",       // int
             "width"  : "",       // int
             "max_intensity": "", // int
             "min_intensity": "", // int
             "hist_intensity": "" // 2d list of float
            }
     },
     //...
     ]
}
``` 
Below code snippet demonstrates the usecase:
```bash
python ImageProcessor.py -i <path/to/TQ1> -m TQ1
```

## TQ2 - Image Segmentation
In this code, [Felzenszwalb’s algorithm](http://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf) is mainly used for the segmentation task. 
In order to run the segmentation, execute the following command:
```bash
python ImageProcessor.py -i <path/to/TQ1> -g .<path/to/TQ/*.tif> -m TQ2
```
The code will generate the following outputs:
|Output|Note|
|---|---|
|TQ2_out.png|Segmentated image|
|Metrics|Console output in JSON format including (1) accuracy, (2) precision, (3) recall, (4) f1, and (5) intersection-over-union|

## TQ3 - Image Analysis
To obtain the features of the segmented objects, run the following command:
```bash
python ImageProcessor.py -i <path/to/TQ1> -m TQ3
```
The output should be stored in JSON format (i.e., `TQ3_out.json`), following the below structure:
```JSON
{
    "mask" : "",            // 2d list of int
    "contours" : [
    {
        "momentum" : "",    // image moments: (float,float)
        "area" : "",        // contour area: float
        "perimeter" : "",   // arc length: float
        "convhull" : "",    // points of polygon: 2d list of (int,int)
        "boundingbox" : "", // coords, width, and height of bb: (int, int, int, int)
        "mincircle" : ""    // coords and radius of minimum enclosing circle: 
                            // (float, float, float]
    },
    //...
    ]
    }
```

## TQ4 - Image Analysis
To detect the signal differences across the series, execute the following command:
```bash
python ImageProcessor.py -i <path/to/TQ4> -m TQ4
```
The code will store a plot image, `TQ4_out.png`. 
The following describes the logic behind the metrics:
First, we compute the absolute differences (i.e., `cv2.abs_diff(frame1,frame2)`) between two consecutive frames. 
Given the absolute differences, we then count the number of pixels greater than a threshold. In particular, a set of thresholds is used (i.e., `[10, 25, 50, 100, 150]`) to capture the degree of differences. That is, the trends of the number of pixels for the relatively small threshold (e.g., `10`) should reflect the differences in low frequency.

## TQ5 - Scaling
In order to scale the solution to process higher-resolution images, I would consider using a `parallel processing` approach. For instance, we can divide the high-resolution image into smaller patches and process the patches in a distributed fashion. To achieve this, I have come to across the library, called [ray](https://www.ray.io/). 
Due to the time constraint, I was not able to optimize the process of generating patches. However, I was able to confirm that the processing time can be significantly reduced with the use of `ray`.
To run the code, execute the following command:
```bash
python ImageProcessor.py -i <path/to/TQ5> -m TQ5
```
The code will generate a set of segmented patches, `TQ5_0_out.png`, `TQ5_1_out.png`, etc.
> **Note .** The patch generation process is not completed. For now, the same patches are statically generated with the fixed size (1024x1024).
> 
## Author
- Chulwoo (Mike) Pack 