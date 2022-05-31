# Hello Applicant
Within this repository, you will find the relevant materials to complete this technical assessment as part of the interview process for TissueVision, Inc.<br>

This challenge is focused on understanding your knowledge of image processing, segmentation, and analysis. The challenge includes both requirements to write model code and documentation for assessing image analysis problems, as well as written answers to broader case questions.<br>

You will have a limited period to complete this challenge and provide your responses. Please provide your responses in a shared Git repository. Please include a `README.md` and all relevant `requirements.txt` files to process your code. Written answers can be provided via `.md` or `.txt` files.


## TQ1 - Image Processing <br>
Included in directory `TQ1/` are 3 representative images acquired from a dataset captured on our TissueCyte Systems: 
- 41077_120219_S000090_ch01.tif
- 41077_120219_S000090_ch02.tif
- 41077_120219_S000090_ch03.tif<br>

Please write example code to extract as much information about the data files as possible. 

## TQ2 - Image Segmentation <br>
Included in directory `TQ2/` is the file *binary_41077_120219_S000090_L01.tif*, which represents a *'ground truth'* binary segmentation mask from the imaging data provided in **TQ1**.<br> 
Please provide example code to:
1. Produce as similar a segmentation as possible using the example data images in **TQ1**.  
2. Evaluate how accurate your segmentation is compared with the *ground truth* provided.

## TQ3 - Image Analysis <br>
For your binary segmentation mask of the data produced in **TQ2**, <br> please provide code to:
1. Describe the features of the segmented objects created from the binary mask. 
2. Highlight the types of filestructures you could use to store the binary mask and object features to maximize read/write. 

## TQ4 - Image Analysis <br>
Included in directory `TQ4/` is a series of images acquired from a dataset captured on our TissueCyte Systems.<br>
What kind of metric would you extract from the images to quantify signal differences across the series of images?<br>

Please provide code to:
1. Extract the metric of signal differences across the series
2. Plot/visualize this metric across the series<br>

Please provide a written response on:<br>

3. How you would validate the metric across the dataset

## TQ5 - Scaling
Please provide the codebase developed above in a packaged form which would allow for running via command line.<br>
- Please include adequate documentaion and a README file for executing the code.
- Please provide your code via a shared Git repository.

Included in directory `TQ5/` are the full resolution images from which the example data in **TQ1** was extracted.<br>
Please provide a written response on:<br>

1. How you would scale your segmentation and signal processing algorithms developed in **TQ2** and **TQ4** to process a series of data consisting of 200 such image sections?
    - Please address input/output requirements, datastructures/libraries utilized, computational limitations, etc...
