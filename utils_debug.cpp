#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <sstream>
#include <iomanip>
#include <iterator>     // std::istream_iterator
#include <fstream>
#include <cstdio>
#include <algorithm>
#include <string>
#include <vector>
#include <cstring>
#include <list>
#include <set>
#include <queue>
#include <iostream>
#include <map>
#include <bitset>
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* wcstombs, wchar_t(C) */
#include <iomanip>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;



void imshow(float*, int, string);
void imshow(vector<float> & v, string title=""){
    imshow(&v[0], (int)sqrt(v.size()), title); 
}
void imshow(float * array, int side, string title=""){
    cout<<"Showing image side "<<side<<endl;
    CvSize size;
    size.height = side ;
    size.width = side;
    IplImage* ipl_image_p = cvCreateImageHeader(size, IPL_DEPTH_32F, 1);
    ipl_image_p->imageData = reinterpret_cast<char*>(array);
    ipl_image_p->imageDataOrigin = ipl_image_p->imageData;
    Mat image(ipl_image_p);
    cout<<"Showing image\n";    
    namedWindow( "ASTEROID"+title, WINDOW_NORMAL );// Create a window for display.
    
    
    imshow( "ASTEROID"+title, image );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
}

void imshow64(double*, int, string);
void imshow64(vector<double> & v, string title=""){
    imshow64(&v[0], (int)sqrt(v.size()), title); 
}
void imshow64(double * array, int side, string title=""){
    cout<<"Showing image side "<<side<<endl;
    CvSize size;
    size.height = side ;
    size.width = side;
    IplImage* ipl_image_p = cvCreateImageHeader(size, IPL_DEPTH_64F, 1);
    ipl_image_p->imageData = reinterpret_cast<char*>(array);
    ipl_image_p->imageDataOrigin = ipl_image_p->imageData;
    Mat image(ipl_image_p);
    cout<<"Showing image\n";    
    namedWindow( "ASTEROID"+title, WINDOW_NORMAL );// Create a window for display.
    
    
    imshow( "ASTEROID"+title, image );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
}
