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

#include <sstream>
#include <iomanip>
#include <iterator>     // std::istream_iterator
    template <typename T>
    T to(const std::string & s)
    {
        std::istringstream stm(s);
        T result;
        stm >> result;
        return result;
    }






    template <typename T>
    string tostr(const T & val)
    {
        std::stringstream stm;
        stm << val;
        return stm.str();
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

    void imshow(vector<float> & img){
    }

    template <typename T>
    void write(T begin, T end)
    {
        return;
    } 
    template <>
    void write(pair<int,int> * begin, pair<int,int> * end){
        return;
    }	

    template<class T, class K>
    ostream& operator<<(ostream& out, const pair<T,K> & p){
        return out;
    }



#endif
using namespace std;


//CONFIG


float im_crop_factor_1 = 2.0f; //before augumentation
float im_crop_factor_2 = 4.0f; //after augumentation
int maximum_pixel_intensity = 255;
int image_side = 64;
int image_side_pre_aug = image_side/(im_crop_factor_1);
int image_side_final = image_side/(im_crop_factor_1*im_crop_factor_2);

//configuring neural network
char * c_nn_data = "0000803f0000004000004040000080400000a0400000c0400000003f0000803e000000400000804000004040000010410000803f000000400000000000000000000000000000803f0000803f0000803f";
vector<int> c_layer_sizes = {3, 2,2};

//CONFIG


class NN{
public:
    vector<vector<float> > Ws, bs;
    vector<float> mean, std;
    vector<int> layers_sizes;
    string activation;
    /*
    * @param hex1_nn_data Encoded in Hex1 data for NN
    */ 
    NN(char* hex1_nn_data,  vector<int> layers_sizes, string activation="linear" ): layers_sizes(layers_sizes), activation(activation) {
        REPORT("Reading neural network");
        int shift = 0;
        vector<float> decoded_nn = decode_hex1(reinterpret_cast<unsigned char*>(hex1_nn_data));
        REPORT("Read "+tostr(decoded_nn.size()));
        //efficient
        for(int i=0;i<layers_sizes.size()-1;++i){
            vector<float> W, b;
            W.resize(layers_sizes[i]*layers_sizes[i+1]);
            b.resize(layers_sizes[i+1]);
            memcpy(&W[0], &decoded_nn[0]+shift, layers_sizes[i]*layers_sizes[i+1]*sizeof(float));
            shift += layers_sizes[i]*layers_sizes[i+1];
            memcpy(&b[0], &decoded_nn[0]+shift, layers_sizes[i+1]*sizeof(float));
            shift += layers_sizes[i+1];
            Ws.push_back(std::move(W));
            bs.push_back(std::move(b));
            write(Ws.back().begin(), Ws.back().end());
            write(bs.back().begin(), bs.back().end());
        } 
        mean.resize(layers_sizes[0]); 
        std.resize(layers_sizes[0]); 
        memcpy(&mean[0], &decoded_nn[0]+shift, layers_sizes[0]*sizeof(float));
        shift += layers_sizes[0];
        memcpy(&std[0], &decoded_nn[0]+shift, layers_sizes[0]*sizeof(float));
        shift += layers_sizes[0];
        //scaling = decode_hex1(scaling_encoded);
    }

    double feedforward(vector<double> input){
        if(input.size() != layers_sizes[0]) throw "Wrong dimensionality for NN";

        //Prepare input
        vector<double> current_input;
        vector<double> previous_input;
        previous_input.reserve(input.size());
        for(auto v: input) previous_input.push_back((double)v);
        
        //Scale
        for(int i=0;i<current_input.size();++i)
            current_input[i] = (current_input[i] - (double)mean[i])/(double)std[i];

        for(int i=0;i<layers_sizes.size()-1;++i){
            current_input.resize(layers_sizes[i+1]);
            for(auto & v: current_input) v = 0.0;
            
            for(int k=0;k<layers_sizes[i+1];++k){
                for(int l=0;l<layers_sizes[i];++l){
                    current_input[k] += (double)Ws[i][k*layers_sizes[i]+l]*previous_input[l];
                }
            }
            for(int k=0;k<layers_sizes[i+1];++k)
                current_input[k] += (double)bs[i][k];
            
            if(activation=="linear"){
            }
            else{
                throw "Not know activation";
            }
                
            write(current_input.begin(), current_input.end());
            previous_input = std::move(current_input);
        }
    }

} g_nn(c_nn_data, c_layer_sizes) ;


// GLOBAL VARIABLES

int g_n_images = 0;
vector<pair<double, int>> g_uuids;

// GLOBAL VARIABLES



void log_transform(vector<int> & img, int offset, vector<double>& out);

#include <cmath>
unsigned int get_side(const vector<double> & img){
    return (int)sqrt((float)img.size());
}

/*
 * Log transform and writes out to out
 * @param offset - where does the image data starts
 */
void log_transform(vector<int> & img, int offset, vector<double>& out){
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

vector<double> im_crop(const vector<double> & img, int side, float factor){
    int cropped_side = side / factor;
    int shift = (side - cropped_side) / 2;
    vector<double> img_cropped(cropped_side*cropped_side, 0);


    for(int i=0;i<cropped_side;++i){
        for(int j=0;j<cropped_side;++j){
            img_cropped[i*cropped_side + j] = img[(i+shift)*side + (j+shift)]; 
        }
    }
    return img_cropped;
}

vector<double> pre_augumentation(vector<int> imageData, int k, float im_crop_factor_pre=im_crop_factor_1){
    vector<double> out(image_side*image_side, 0.0);
    log_transform(imageData, k*image_side*image_side, out);
    for(auto &v: out) v /= ((float)maximum_pixel_intensity);
    vector<double> img_cropped = im_crop(out, get_side(out), im_crop_factor_pre);
    return img_cropped;
}

vector<double> transform_image(const vector<double> &img, float im_crop_factor_post = im_crop_factor_2){
    //REPORT("augumentation");
    //REPORT("cropping");
    vector<double> im_cropped = im_crop(img, get_side(img), im_crop_factor_post);
    return im_cropped;
}


    
/* Prepare data for i-th image (0-1-2-3 frame 0-1-2-3 frame etc...)
 * @returns vector<float>
 * @param k - kth image
 */
vector<double> prepare_data(vector<int> & imageData, vector<string> & metadata, int k){
    REPORT("transforming image");
    vector<double> img = pre_augumentation(imageData, k);
    vector<double> augumented_img = transform_image(img);
    int ExtraColumns = 6;
    
    augumented_img.resize(augumented_img.size() + ExtraColumns);
    augumented_img.push_back(to<double>(metadata[9]));
    augumented_img.push_back(to<double>(metadata[10]));
    augumented_img.push_back(to<double>(metadata[11]));
    augumented_img.push_back(to<double>(metadata[12]));
    augumented_img.push_back(to<double>(metadata[13]));
    augumented_img.push_back(to<double>(metadata[14]));
    
    return augumented_img;
}






/*
 * 4 patches of 64x64 at 4 different observation times
 *
 * @param imageData - 1d array containing info i*64*64 + x + y*64
 * 
 * 
 */
int trainingData(vector<int> imageData, vector<string> detections){

    return 0; 
}
/*
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
}*/

/*
 * 4 patches of 64x64 at 4 different observation times
 *
 * @param imageData - 1d array containing info i*64*64 + x + y*64
 * 
 * 
 */
int testingData(vector<int> imageData, vector<string> detections){
    
    vector<vector<string> > metadata;

    REPORT("Training");
    int n = imageData.size() /( image_side*image_side);
    int k = n/4;
   
    g_n_images += n;
     
    COUT("Training data got ");
    REPORT(n);
    REPORT(k);

    REPORT(detections.size());
    

    REPORT("done");
   
    for(int i=0;i<n;++i){
       istringstream iss(detections[i]);
       vector<string> tokens{istream_iterator<string>{iss},
             istream_iterator<string>{}};
       metadata.push_back(tokens);
    }
    for(int i=0;i<k;++i){
       g_uuids.push_back(pair<double, int>(1.0f, to<int>(metadata[4*i][0])));
    }
    //Get neural network input
    vector<double> input = prepare_data(imageData, metadata[0], 0);

    REPORT(input.size());

    //Get image from that
    vector<double> img(input.begin(), input.begin()+image_side_final*image_side_final);
    imshow64(input);   
 
    return 0;
}




vector<pair<double, int>> getAnswer(){
    sort(g_uuids.begin(), g_uuids.end());
    return g_uuids;
}

int main(){

   
    int N, M,i,j;
    string tmp;
    for (i=0; i < 1; i++)
    {
        vector<int> imageData;
        vector<string> detections;
        cin>>N;
        imageData.resize(N);
        for (j=0; j < N; j++)
            cin >> imageData[j];
        tmp="";
        cin>>M;    
        detections.resize(M);
        getline(cin, tmp);
        for (j=0; j < M; j++){
            getline(cin, detections[j]);
            REPORT(detections[j]);
        }
        REPORT(i);
        REPORT(N);
        REPORT(M);
        int result = trainingData(imageData, detections);
        cout<<result<<endl;
        cout<<flush;
    }
    for (i=0; i < 1; i++)
    {
        vector<int> imageData;
        vector<string> detections;
        cin>>N;
        REPORT(N);
        imageData.resize(N);
        REPORT(N);
        for (j=0; j < N; j++)
            cin >> imageData[j];
        cin>>M;
        detections.resize(M);
        getline(cin, tmp);
        for (j=0; j < M; j++)
            getline(cin, detections[j]);
        int result = testingData(imageData, detections);
        cout<<result<<endl;
        cout<<flush;
    }
    REPORT("flushing results");
    auto results = getAnswer();
    cout<<results.size()<<endl;
    for (i=0;i < results.size(); i++)
        cout<<results[i].second<<endl;
    cout<<flush;
}

