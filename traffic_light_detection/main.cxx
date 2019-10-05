#include <iostream>  
#include <vector>  
#include <cmath>
#include <cstdio>
#include <cstring>
#include "recognition.hpp"
#include <malloc.h>
using namespace std;
using namespace sidewalkTrafficLight;

int main(int argc, char *argv[])
{
	recognition recognize = recognition();
	for( int i = 1; i<argc; i++){
		int status = recognize.processGraph(string(argv[i]));
		if(status == GREEN) cout<<"Green detected"<<endl;
		else if(status == RED) cout<<"Red detected"<<endl;
		else cout<<"Background detected"<<endl;
	}
	return 0;
}