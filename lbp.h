/*
 * lbp.h.
 *
 * Written by: Pascal Mettes.
 *
 * This file contains the class for the NN image classification algorithm using
 * LBP histograms.
 */
 
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
using namespace std;
using namespace cv;

/* Convenient definition for 2d vector. */
#define vec2dd vector<vector<double> >
 
/* Spatial stride used when going through all the pixels in each image. */
#define STRIDE 3

/*
 * This class contains all the variables and functions for a NN image 
 * classification algorithm based on standard LBP histograms.
 */
class LBPClassifier {
    public:
        /* Constructor and descructor. */
        LBPClassifier(string trainfile, string testfile, int p, double r, int c);
        ~LBPClassifier();
        
        /* Functions for training and testing LBP histograms. */
        void train();
        void test();
    
    protected:
        /* Container information for the train set. */
        vector<string> trainimages;
        vector<int> trainlabels;
        /* Container information for the test set. */
        vector<string> testimages;
        vector<int> testlabels;
        
        /* The number of neighbouring pixels to comare. */
        int p;
        /* The radius between the center pixel and its neighbours. */
        double r;
        
        /* The number of target classes. */
        int c;
        
        /* The train histograms (set in train() function. */
        vec2dd histograms;
};
