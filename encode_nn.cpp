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

#ifdef DEBUG
    #include<stdlib.h>

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




unsigned char * encode_hex1(string filename){
    vector<unsigned char> *  encoded_ptr = new vector<unsigned char>();
    vector<unsigned char> & encoded = *encoded_ptr;
    ifstream myFile (filename, ios::in );
    vector<float> float_array;
	//float_array.push_back(0.551420390606);    
	copy(istream_iterator<float>(myFile), istream_iterator<float>(), back_inserter(float_array));
    for(auto v: float_array){
        unsigned char * v_bytes = reinterpret_cast<unsigned char*>(&v);
        //Go through each byte
        for(int i=0;i<4;++i){
            stringstream ss;
            //Get hex 
            ss<<hex<<(int)v_bytes[i];
            string encoded_hex = ss.str();

            
            if(encoded_hex.length()>1){
                encoded.push_back(encoded_hex[0]);
				encoded.push_back(encoded_hex[1]);
			}
            else{
                encoded.push_back('0');
				encoded.push_back(encoded_hex[0]);
			}
        }
    }
    encoded.push_back('\0');
    for(auto x: encoded) cout<<x;
    myFile.close();
    return &(encoded[0]);
}



int main(int argc, char ** argv){
    unsigned char * hex1in = encode_hex1(argv[1]);
    vector<float> hex1in_floats = decode_hex1(hex1in);
 	//write(hex1in_floats.begin(), hex1in_floats.end());

}
