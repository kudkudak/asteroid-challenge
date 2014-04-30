


#define DEBUG
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



//CONFIG
//

float im_crop_factor = 2.0f;
int maximum_pixel_intensity = 255;
int image_side = 64;
int image_side_pre_crop = 32;
int image_side_final = 16;

//
//CONFIG



// GLOBAL VARIABLES

int g_n_images = 0;
vector<int> uuids;

// GLOBAL VARIABLES





using namespace std;

typedef vector<int> VI;
typedef long long LL;
typedef pair<int,int> PI;
typedef pair<double,double> PD;

#define FOR(x, b, e) for(int x=b; x<=(e); ++x)
#define FORD(x, b, e) for(int x=b; x>=(e); ......x)
#define REP(x, n) for(int x=0; x<(n); ++x)
#define VAR(v,n) typeof(n) v=(n)
#define ALL(c) c.begin(),c.end()
#define SIZE(x) (int)(x).size()
#define FOREACH(i,c) for(VAR(i,(c).begin());i!=(c).end();++i)
#define PB push_back
#define ST first
#define ND second
#define INF 10000000; 



#ifdef DEBUG
    #include<stdlib.h>
    #include "utils_debug.cpp"

	#define NEWLINE cout<<"\n"
	#define REPORT(x) cout<<#x<<"="<<(x)<<endl<<flush;
    #define COUT(x) cout<<(x)<<endl;
	#define ASSERT(x) \
            if(!(x)){ \
                 REPORT("warunek nie spelniony!!");\
                 exit(1);\
            }

    template <typename T>
    void write(T begin, T end)
    {
        T ptr = begin;
        while(ptr!=end){
            cout<<*(ptr++)<<" ";
        }
        cout<<endl;
    } 
    template <>
    void write(pair<int,int> * begin, pair<int,int> * end){
        pair<int,int>* ptr = begin;
        while(ptr!=end){
            cout<<"("<<(ptr->first)<<","<<(ptr->second)<<") ";
            ++ptr;
        }
        cout<<endl;

    }	
    template <typename T>
    T to(const std::string & s)
    {
        std::istringstream stm(s);
        T result;
        stm >> result;
        return result;
    }



    template<class T, class K>
    ostream& operator<<(ostream& out, const pair<T,K> & p){
        out<<"("<<p.first<<","<<p.second<<")";
        return out;
    }


#else
    #define REPORT(x) ;
    #define COUT(x) ;
    #define NEWLINE ;
    #define ASSERT(x) ;

    template <typename T>
    void write(T begin, T end)
    {
        return;
    } 
    template <>
    void write(pair<int,int> * begin, pair<int,int> * end){
        return;
    }	
    template <typename T>
    T to(const std::string & s)
    {
        return;
    }
    template<class T, class K>
    ostream& operator<<(ostream& out, const pair<T,K> & p){
        return out;
    }



#endif
using namespace std;



std::string dec2bin(unsigned n){
    const int size=sizeof(n)*8;
    std::string res;
    bool s=0;
    for (int a=0;a<size;a++){
        bool bit=n>>(size-1);
        if (bit)
            s=1;
        if (s)
            res.push_back(bit+'0');
        n<<=1;
    }
    if (!res.size())
        res.push_back('0');
    return res;
}

#include <sstream>
#include <iomanip>
#include <iterator>     // std::istream_iterator
unsigned char * encode_hex1(string filename){
    vector<unsigned char> *  encoded_ptr = new vector<unsigned char>();
    vector<unsigned char> & encoded = *encoded_ptr;
    ifstream myFile (filename, ios::in );
    vector<float> float_array;
    copy(istream_iterator<float>(myFile), istream_iterator<float>(), back_inserter(float_array));
    for(auto v: float_array){
        unsigned char * v_bytes = reinterpret_cast<unsigned char*>(&v);
        //Go through each byte
        for(int i=0;i<4;++i){
            stringstream ss;
            //Get hex 
            ss<<hex<<(int)v_bytes[i];
            string encoded_hex = ss.str();
            encoded.push_back(encoded_hex[0]);
            if(encoded_hex.length()>1)
                encoded.push_back(encoded_hex[1]);
            else
                encoded.push_back('0');
        }
    }
    encoded.push_back('\0');
    write(encoded.begin(), encoded.end());
    cout<<encoded[0]<<" "<<encoded[1]<<endl;
    write(float_array.begin(), float_array.end());
    myFile.close();
    return &(encoded[0]);
}

vector<float> decode_hex1(unsigned char * hex1){
    vector<float> decoded;
    //Build floats
    unsigned char built_float[4];
    char built_hex[3];
    float float_memholder;
    int current = 0;
    for(int i=0;i<strlen((char*)hex1)/2;++i){
        //Build hex number
        built_hex[0] = hex1[2*i];
        built_hex[1] = hex1[2*i+1];
        built_hex[2] = '\0';
        //Add to build_float
        built_float[current] = (int)strtol(built_hex, NULL, 16);
        current += 1;
        if(current == 4){
            current = 0;
            memcpy(&float_memholder, reinterpret_cast<float*>(built_float), sizeof(float));  
            decoded.push_back(float_memholder);
        }
    }
    return decoded; 
}

#include <cmath>
unsigned int get_side(const vector<float> & img){
    return (int)sqrt((float)img.size());
}



vector<float> im_crop(const vector<float> & img, int side, float factor){
    int cropped_side = side / factor;
    int shift = (side - cropped_side) / 2;
    vector<float> img_cropped(cropped_side*cropped_side, 0);


    for(int i=0;i<cropped_side;++i){
        for(int j=0;j<cropped_side;++j){
            img_cropped[i*cropped_side + j] = img[(i+shift)*side + (j+shift)]; 
        }
    }
    return img_cropped;
}

vector<float> transform_image(const vector<float> &img, float im_crop_factor_local = im_crop_factor){
    vector<float> im_cropped = im_crop(img, get_side(img), im_crop_factor_local);
    
    for(float &v: im_cropped) v /= ((float)maximum_pixel_intensity);
    return im_cropped;
}

/*
 * Log transform and writes out to out
 * @param offset - where does the image data starts
 */
void log_transform(vector<int> & img, int offset, vector<float>& out){
    int W = (64+10)*4-10;
    int out_id = 0;
    int off = offset;
    int imin = 1<<20;
    int imax = -imin;
    // Find min and max
    for (int j=0;j<4096;j++)
    {
        int r = img[j+off];
        if (r>65500) continue;
        imin = min(imin, r);
        imax = max(imax, r);
    }
    double dmax = (double)(imax) / 256.0;
    double dmin = (double)(imin) / 256.0;
    if (dmax*0.5-dmin > 10)
    {
        dmax *= 0.5;
    }
    if (dmax-dmin<0.0001) dmax += 0.1;

    double linearF = 255.0 / (dmax - dmin);
    double log10 = log(10.0);
    double logF = 255.0 / (log(255.0) / log10);
    for (int y=0;y<64;y++)
    for (int x=0;x<64;x++)
    {
        double ival = (double)img[off++];
        double dval = (double)(ival) / 256.0;
        if (dval<dmin) ival = 0;
        else if (dval>dmax) ival = 255;
        else
        {
            dval = max(0.0, min(dval-dmin, dmax - dmin));
            double d = ((log(dval * linearF)) / log10) * logF;
            ival = d;
        }
        if (ival<0) ival = 0;
        if (ival>255) ival = 255;
        out[out_id++]=ival;
    }
}
    
/* Prepare data for i-th image (0-1-2-3 frame 0-1-2-3 frame etc...)
 * @returns vector<float>
 * @param k - kth image
 */
vector<float> prepare_data(vector<int> & imageData, vector<string> & detections, int k){
    vector<float> out(image_side_pre_crop*image_side_pre_crop, 0.0);
    log_transform(imageData, k*image_side*image_side, out);
    return transform_image(out, im_crop_factor);
}


/*
 * 4 patches of 64x64 at 4 different observation times
 *
 * @param imageData - 1d array containing info i*64*64 + x + y*64
 * 
 * 
 */
int trainingData(vector<int> imageData, vector<string> detections){
    int n = imageData.size() / image_side*image_side;
    int k = n/4;
   
    g_n_images += n;
     
    COUT("Training data got ");
    REPORT(n);
    REPORT(k);

    vector<float> input = prepare_data(imageData, detections, 0);
    imshow(input);


}

void test(){

    unsigned char * hex1in = encode_hex1("2.in");
    vector<float> hex1in_floats = decode_hex1(hex1in);
    write(hex1in_floats.begin(), hex1in_floats.end());
    delete[] hex1in;

    vector<float> converted2 = transform_image(hex1in_floats, 1.0);
    imshow(converted2);
    //imshow(&im_crop(test_show, get_side(test_show), 2.0)[0], 50);
    vector<float> converted = transform_image(hex1in_floats);
    for(float &v: converted) cout<<v<<" ";
    cout<<endl;

    imshow(converted);
}

/*
 * 4 patches of 64x64 at 4 different observation times
 *
 * @param imageData - 1d array containing info i*64*64 + x + y*64
 * 
 * 
 */
int testingData(vector<int> imageData, vector<string> detections){
    return 1;
}




vector<int> getAnswer(){
    return {0};
}

int main(){
    
    int N, M,i,j;
    vector<int> imageData;
    vector<string> detections;
    for (i=0; i < 1000; i++)
    {
        cin >> N;
        imageData.resize(N);
        for (j=0; j < N; j++)
            cin >> imageData[j];
        cin >> M;
        for (j=0; j < M; j++)
            cin >> detections[j];
        int result = trainingData(imageData, detections);
        cout<<result<<endl;
        cout<<flush;
    }
    for (i=0; i < 200; i++)
    {
        cin >> N;
        imageData.resize(N);
        for (j=0; j < N; j++)
            cin >> imageData[j];
        cin >> M;
        for (j=0; j < M; j++)
            cin >> detections[j];
        int result = testingData(imageData, detections);
        cout<<result<<endl;
        cout<<flush;
    }
    vector<int> results = getAnswer();
    cout<<results.size()<<endl;
    for (i=0;i < g_n_images; i++)
        cout<<results[i]<<endl;
    cout<<flush;
}

