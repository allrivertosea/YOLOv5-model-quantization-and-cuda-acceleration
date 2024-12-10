#include <iostream>

#include <string>
#include <vector>
#include "assert.h"
#include <time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace std;
class CocoLabels {
public:
    CocoLabels() {
        for(int i = 0 ; i < 8; i ++) {
            std::string x = "NA";
            mLabels.push_back( x );
        }

        mLabels[0]  = "car";
        mLabels[1]  = "truck";
        mLabels[2]  = "person";
        mLabels[3]  = "bicycle";
        mLabels[4]  = "cyclist";
        mLabels[5]  = "van";
        mLabels[6]  = "tricycle";
        mLabels[7]  = "bus";

    }

    std::string coco_get_label(int i) {
        assert( i >= 0 && i < 8 );
        return mLabels[i];
    }

    cv::Scalar coco_get_color(int i) {
        float r;
        srand(i);
        r = (float)rand() / RAND_MAX;
        int red    = int(r * 255);

        srand(i + 1);
        r = (float)rand() / RAND_MAX;
        int green    = int(r * 255);

        srand(i + 2);
        r = (float)rand() / RAND_MAX;
        int blue    = int(r * 255);

        return cv::Scalar(blue, green, red);
    }

    cv::Scalar get_inverse_color(cv::Scalar color) {
        int blue = 255 - color[0];
        int green = 255 - color[1];
        int red = 255 - color[2];
        return cv::Scalar(blue, green, red);
    }


private:
  std::vector<string> mLabels;

};
