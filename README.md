# GORP Counting with Mean Shift

Project completed in CS 283 at Harvard by Alex Wilentz and Joshua Price. 

![GORP](fig/Fig3.jpeg?raw=true "Title")

To run the modified mean shift code, simply run `python3 classification_subset.py` in this project's directory.

There are four global variables to adjust what happens when you run the file:
1. BANDWIDTH: We typically use 40-50 for our GORP images.
2. SUBSET_SIZE: Size of sample for mean shift. We recommend >4000. The algorithm will substantially slow with higher sample sizes.
3. CUSTOM: If True, our implementation will be run. If False, sklearn's will be run.
4. IMAGE_PATH: Path to image you would like to analyze.

If you don't change any code, it will generate a figure showing the segmentation and print out the number of classes identified and counts of objects assigned to each class.

For questions, email awilentz@college.harvard.edu and/or joshuaprice@g.harvard.edu.
