/*
 * lbp.cpp.
 *
 * Written by: Pascal Mettes.
 */

#include "lbp.h"

/*
 * Load a set specified by a file containing the labels and image locations.
 *
 * Input : The filename (string) and references to the set of image locations
 *         (vector<string>) and labels (vector<int>).
 * Output: -
 */
void load_set(string filename, vector<string> &images, vector<int> &labels) {
    ifstream imagefile(filename.c_str());
    string line, key, value;

    /* Go through each line. */
    while (getline(imagefile, line)) {
        istringstream lstream(line);

        /* Add the label and image location to the list. */
        if (getline(lstream, key, ' ')) {
            if (getline(lstream, value, ' ')) {
                labels.push_back(atoi(key.c_str()));
                images.push_back(value);
            }
        }
    }
    imagefile.close();
}

/*
 * Compute the norm of a list of values.
 *
 * Input : The list of values (vector<double>).
 8 Output: The norm (double).
 */
double norm(vector<double> histogram) {
    double total = 0.0;
    
    for (int i = 0; i < (int) histogram.size(); i++) {
        total += histogram[i];
    }
    
    return total;
}

/*
 * Perform nearest neighbour search on a test histogram given a set of train
 * histograms.
 *
 * Input : The test histogram (vector<double>) and train histograms (vec2dd).
 * Output: The index of the nearest train neighbour (int).
 */
int nn_search(vector<double> hist, vec2dd histograms) {
    double mindist = DBL_MAX;
    int index = 0;
    
    for (int i = 0; i < (int) histograms.size(); i++) {
        double d = 0.0;
        
        for (int j = 0; j < (int) histograms[i].size(); j++) {
            d += pow(hist[j] - histograms[i][j], 2);
        }
        d = sqrt(d);
        
        if (d < mindist) {
            mindist = d;
            index = i;
        }
    }
    
    return index;
}

/*
 * Compute the LBP value for a single pixel.
 *
 * Input : The image (Mat), the x-value (int), the y-value (int), the number
 *         of pixel in the comparison (int), and the radius (double).
 * Output: The LBP value (int).
 */
int lbp(Mat image, int x, int y, int p, double r) {
    assert(p > 0 && r > 0);

    int center = (int) image.at<uchar>(y, x);
    int value = 0;
    
    for (int i = 0; i < p; i++) {
        /* Update the offset wrt the center pixel. */
        double dx = sin(i / ((double)p) * 2 * M_PI) * r;
        double dy = cos(i / ((double)p) * 2 * M_PI) * r;
    
        /* Compare and update LBP value. */
        int element = (int) image.at<uchar>(round(y + dy), round(x + dx));
        value += (element >= center) * pow(2,i);
    }
    
    return value;
}

/*
 * Compute a histogram of LBP values for an image.
 *
 * Input : The image (Mat), the number of pixel in the comparison (int), and
 *         the radius (double).
 * Output: The LBP histogram (vector<double>.
 */
vector<double> compute_lbp_histogram(Mat image, int p, double r) {
    vector<double> histogram((int)pow(2,p), 0.0);
    
    /* Go through all the pixels. */
    for (int x = ceil(r); x < image.cols - ceil(r); x += STRIDE) {
        for (int y = ceil(r); y < image.rows - ceil(r); y += STRIDE) {
            int index = lbp(image, x, y, p, r);
            histogram[index] += 1;
        }
    }
    
    /* Normalize the histogram. */
    double total = norm(histogram);
    for (int i = 0; i < (int) histogram.size(); i++) {
        histogram[i] /= total;
    }
    
    return histogram;
}

/*
 * Initialize the classifier.
 *
 * Input : The trainfile (string), the test file (string), the number of points
 *         to compare each pixel (int), the radius (double), and the number of
 *         target classes (int).
 * Output: -
 */
LBPClassifier::LBPClassifier(string trainfile, string testfile, int p, double r, int c) {
    /* Load the train and test sets. */
    load_set(trainfile, trainimages, trainlabels);
    load_set(testfile, testimages, testlabels);
    
    /* Set the number of pixels and radius. */
    this->p = p;
    this->r = r;
    this->c = c;
}

/*
 * Do nothing extra at descruction.
 *
 * Input : -
 * Output: -
 */
LBPClassifier::~LBPClassifier() {

}

/*
 * Compute the histograms of the train images.
 *
 * Input : -
 * Output: -
 */
void LBPClassifier::train() {
    for (int i = 0; i < (int) trainimages.size(); i++) {
        printf("Extracting LBP histogram for train image %d/%d\r", i+1, (int) trainimages.size());
        fflush(stdout);
        
        Mat image = imread(trainimages[i], CV_LOAD_IMAGE_GRAYSCALE);
        histograms.push_back(compute_lbp_histogram(image, p, r));
        image.release();
    }
    printf("\n");
}

/*
 * Compute the histograms of the test images and perform classification using
 * the train histograms.
 *
 * Input : -
 * Output: -
 */
void LBPClassifier::test() {
    assert(histograms.size() > 0);

    /* Keep track of the classification rates per target class. */
    vector<double> cclass(c, 0.0);
    vector<double> ccorrect(c, 0.0);
    
    for (int i = 0; i < (int) trainimages.size(); i++) {
        printf("Extracting LBP histogram for test image %d/%d\r", i+1, (int) testimages.size());
        fflush(stdout);
        
        /* Load the image and compute the LBP histogram. */
        Mat image = imread(testimages[i], CV_LOAD_IMAGE_GRAYSCALE);
        vector<double> histogram = compute_lbp_histogram(image, p, r);
        image.release();
        
        /* Perform NN search on the train histograms. */
        int nn = nn_search(histogram, histograms);
        
        /* Update classification rate. */
        cclass[testlabels[i]] += 1;
        ccorrect[testlabels[i]] += (testlabels[i] == trainlabels[nn]);
    }
    printf("\n\n");
    
    for(int i = 0; i < c; i++) {
        printf("Class %d: %d/%d\n", i, (int) ccorrect[i], (int) cclass[i]);
    }
    printf("\nTotal: %d/%d = %f\n", (int)norm(ccorrect), (int)norm(cclass), norm(ccorrect) / norm(cclass));
}

/*
 * Starting point of the algorithm.
 *
 * Input : -
 * Output: -
 */
int main(int argc, char *argv[]) {
    assert(argc == 3);

    LBPClassifier classifier(argv[1], argv[2], 8, 1.0, 10);

    classifier.train();
    classifier.test();

    return 0;
}
