# Digit_Reader
A low level neural network implementation that can interpret hand written images through pixel intensity.

This projects uses a simple neural network implementation in order to train itself to recognize handwritten digits. 
The project is trained from the mnist database, using stochastic gradient descent as well as backdrop in order to
quickly learn.

The program then tests the results of the trained neural network against a test set of handwritten digits. The results of these tests
are there printed to the console. The program is able to reach 95.6% accuracy, and with improvements will be able to hit much higher.

**Creating Project**

When cloning the repository it is important that you clone it recursively using "git clone --recursive" as the project depends on the Eigen package.

**Building Project**

In order to build the project, cmake is required. Navigate to the folder conataining the project and run "cmake build".

**Running Project**

The first thing the program will ask is what inputs to use. Theses can be found in the inputs folder. Enter the filename of the inputs you would like to use from the inputs folder and the program will begin to run.

**Results**
After the file has ran through the number of epochs requested, the results are saved in the file results.csv, in the Results directory. These result allow you to use the effectiveness reached in a less labor intensive way.
