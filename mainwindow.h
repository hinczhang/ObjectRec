#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include<iostream>
#include<fstream>
#include<opencv2/imgproc.hpp>
#include<opencv2/ml.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/dnn.hpp>
#include<opencv2/dnn/dnn.hpp>
#include<qgraphicsitem.h>
#include<qgraphicsscene.h>
#include<QFileDialog>
#include<QMessageBox>
using namespace cv;
using namespace ml;
using namespace std;
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
public slots:
    void videoOpen();
    void videoClose();
    void loadImage();
    void Confirm();
private:
    Ui::MainWindow *ui;
    QImage Mat2QImage(const Mat &mat);
    vector<cv::String> getOutputNames(const cv::dnn::Net& net);
    void postprocess(cv::Mat& frame,const vector<cv::Mat>& outs,int *peoplenum);
    void postprocess(cv::Mat& frame,const vector<cv::Mat>& outs);
    void drawPred(int classId,float conf,int left,int top,int right,int bottom,cv::Mat& frame);
    vector<string>classes;
    cv::dnn::Net net;
    bool flag;
    bool cvFlag;
    VideoCapture cap;
    Mat frame,myframe;
    int num;
    float confThreshold;//置信度阈值
    float nmsThreshold;//非最大抑制阈值
    int inpWidth;//网络输入图片宽度
    int inpHeight;//网络输入图片高度

    QGraphicsScene *pScence;
    QGraphicsPixmapItem *pixItem;
};

#endif // MAINWINDOW_H
