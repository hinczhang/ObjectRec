#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    flag=false;
    cvFlag=false;
    num=0;

    confThreshold = 0.5;//置信度阈值
    nmsThreshold = 0.4;//非最大抑制阈值
    inpWidth = 416;//网络输入图片宽度
    inpHeight = 416;//网络输入图片高度

    string classesFile = "C:\\Users\\23501\\Documents\\QT Projects\\build-YoloDetection-Desktop_Qt_5_11_0_MSVC2017_64bit-Debug\\debug\\coco.names";//coco.names包含80种不同的类名
    ifstream ifs(classesFile.c_str());
    string line;
    while(getline(ifs,line))classes.push_back(line);
    //取得模型的配置和权重文件

    //加载网络
    String cfg="C:\\Users\\23501\\Documents\\QT Projects\\build-YoloDetection-Desktop_Qt_5_11_0_MSVC2017_64bit-Debug\\debug\\yolov3.cfg";
    String weights="C:\\Users\\23501\\Documents\\QT Projects\\build-YoloDetection-Desktop_Qt_5_11_0_MSVC2017_64bit-Debug\\debug\\yolov3.weights";

    ifstream a,b;
    a.open(cfg.c_str());
    b.open(weights.c_str());
    int k1=ifs.is_open();
    int k2=a.is_open();
    int k3=b.is_open();


    net = cv::dnn::readNetFromDarknet(cfg,weights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
    //net.setPreferableBackend(cv::dnn::DNN_TARGET_OPENCL);
    pScence=new QGraphicsScene;
    pixItem=new QGraphicsPixmapItem;
    pScence->addItem(pixItem);
    ui->graphicsView->setScene(pScence);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::videoOpen()
{

    cvFlag=true;
    cap.open(0);
    Mat blob;
    while (cv::waitKey(1) < 0) {
    //取每帧图像
        cap >> myframe;
        //如果视频播放完则停止程序
        if (myframe.empty()) {
            break;
         }
        //在dnn中从磁盘加载图片
        cv::dnn::blobFromImage(myframe, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight));
        //设置输入
        net.setInput(blob);
        //设置输出层
        vector<cv::Mat> outs;//储存识别结果
        net.forward(outs, getOutputNames(net));
        //移除低置信度边界框
        postprocess(myframe, outs);
        //显示s延时信息并绘制
        vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = cv::format("Infercence time for a myframe:%.2f ms", t);
        cv::putText(myframe, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
        //绘制识别框
        QPixmap pix=QPixmap::fromImage(Mat2QImage(myframe));
        pixItem->setPixmap(pix);
        ui->graphicsView->repaint();
       }

}

void MainWindow::videoClose()
{
if(cvFlag==true)
   {
    cvFlag=false;
    putText(myframe,"STOP",Point(myframe.cols/2*1.5,30),FONT_HERSHEY_COMPLEX
            ,1,Scalar(0,0,255),1);
    QPixmap pix=QPixmap::fromImage(Mat2QImage(myframe));
    pixItem->setPixmap(pix);
    ui->graphicsView->repaint();
    cap.release();
}
else
    QMessageBox::information(NULL, "Error", "No Camera is Working!",
                         QMessageBox::Yes);
}

void MainWindow::loadImage()
{
    if(cvFlag==true){
        cap.release();
        cvFlag=false;
    }
    QString fileName;
    fileName = QFileDialog::getOpenFileName(this,tr("Load File"),"F:",tr("image(*png *jpg *tif);;"
                    "test(*txt)"));
    frame=imread(fileName.toLatin1().data());

    if (!frame.empty()){
        QMessageBox::information(NULL, "Loading Result", "Succeed to load the file!",QMessageBox::Yes);
        flag=true;
        QPixmap pix=QPixmap::fromImage(Mat2QImage(frame));
        pixItem->setPixmap(pix);
        ui->graphicsView->repaint();
        return;
         }
    else{
        QMessageBox::information(NULL, "Loading Result", "Error 101: Fail to load the file!\nPlease check the type of the file or the path of the file.",
               QMessageBox::Yes);
        return;
    }


}

void MainWindow::Confirm()
{
    if(flag==true){
        QString sigma;
        sigma=ui->lineEdit->text();
        if(sigma.isEmpty()){
            QMessageBox::information(NULL, "Error", "Invalid Input!",
                                     QMessageBox::Yes);
            return;

        }


        num=ui->lineEdit->text().toInt();
        if(num==0){
            QMessageBox::information(NULL, "Error", "Invalid Input or Zero Input!",
                                     QMessageBox::Yes);
            return;
        }
        Mat blob;
        dnn::blobFromImage(frame,blob,1/255.0,Size(inpWidth,inpHeight));
        net.setInput(blob);
        vector<Mat>outs;
        net.forward(outs,getOutputNames(net));
        int postnum=0;
        Mat temp=frame.clone();
        postprocess(temp,outs,&postnum);

        if(postnum>num||postnum==0)
            QMessageBox::information(NULL, "Report", "There are "+QString::fromStdString(to_string(postnum))+" people in the picture!\nIt's a Scene Picture",
                                 QMessageBox::Yes);
        else
            QMessageBox::information(NULL, "Report", "There are "+QString::fromStdString(to_string(postnum))+" people in the picture!\nIt's a Character Picture",
                                 QMessageBox::Yes);
        QPixmap pix=QPixmap::fromImage(Mat2QImage(temp));
        pixItem->setPixmap(pix);
        ui->graphicsView->repaint();

    }
    else{
        QMessageBox::information(NULL, "Error", "No Image Loaded!",
                             QMessageBox::Yes);
    }
}

QImage MainWindow::Mat2QImage(const Mat &mat)
{
    switch ( mat.type() )
          {
             // 8-bit, 4 channel
             case CV_8UC4:
             {
                QImage image( mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB32 );
                return image;
             }

             // 8-bit, 3 channel
             case CV_8UC3:
             {
                QImage image( mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888 );
                return image.rgbSwapped();
             }

             // 8-bit, 1 channel
             case CV_8UC1:
             {
                static QVector<QRgb>  sColorTable;
                // only create our color table once
                if ( sColorTable.isEmpty() )
                {
                   for ( int i = 0; i < 256; ++i )
                      sColorTable.push_back( qRgb( i, i, i ) );
                }
                QImage image( mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8 );
                image.setColorTable( sColorTable );
                return image;
             }

             default:
                qDebug("Image format is not supported: depth=%d and %d channels\n", mat.depth(), mat.channels());
                break;
          }
          return QImage();
}

vector<String> MainWindow::getOutputNames(const dnn::dnn4_v20180917::Net &net)
{
    static vector<cv::String> names;
        if(names.empty()){
            //取得输出层指标
            vector<int> outLayers = net.getUnconnectedOutLayers();
            vector<cv::String> layersNames = net.getLayerNames();
            //取得输出层名字
            names.resize(outLayers.size());
            for(size_t i =0;i<outLayers.size();i++){
                names[i] = layersNames[outLayers[i]-1];
            }
        }
        return names;
}

void MainWindow::postprocess(Mat &frame, const vector<Mat> &outs,int *peoplenum)
{
    vector<int> classIds;//储存识别类的索引
        vector<float> confidences;//储存置信度
        vector<cv::Rect> boxes;//储存边框
        for(size_t i=0;i<outs.size();i++){
        //从网络输出中扫描所有边界框
        //保留高置信度选框
        //目标数据data:x,y,w,h为百分比，x,y为目标中心点坐标
            float* data = (float*)outs[i].data;
            for(int j=0;j<outs[i].rows;j++,data+=outs[i].cols){
                cv::Mat scores = outs[i].row(j).colRange(5,outs[i].cols);
                cv::Point classIdPoint;
                double confidence;//置信度
                //取得最大分数值与索引
                cv::minMaxLoc(scores,0,&confidence,0,&classIdPoint);
                if(confidence>confThreshold){
                    int centerX = (int)(data[0]*frame.cols);
                    int centerY = (int)(data[1]*frame.rows);
                    int width = (int)(data[2]*frame.cols);
                    int height = (int)(data[3]*frame.rows);
                    int left = centerX-width/2;
                    int top = centerY-height/2;
                    classIds.push_back(classIdPoint.x);
                           confidences.push_back((float)confidence);
                           boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
        //低置信度
        vector<int> indices;//保存没有重叠边框的索引
        //该函数用于抑制重叠边框
        cv::dnn::NMSBoxes(boxes,confidences,confThreshold,nmsThreshold,indices);
        for(size_t i=0;i<indices.size();i++){
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            if(classIds[idx]==0){
            drawPred(classIds[idx],confidences[idx],box.x,box.y,
            box.x+box.width,box.y+box.height,frame);}
        }
        for(int i=0;i<classIds.size();i++){
            if(classIds[i]==0)
            (*peoplenum)++;
        }

}

void MainWindow::postprocess(Mat &frame, const vector<Mat> &outs)
{
    vector<int> classIds;//储存识别类的索引
        vector<float> confidences;//储存置信度
        vector<cv::Rect> boxes;//储存边框
        for(size_t i=0;i<outs.size();i++){
        //从网络输出中扫描所有边界框
        //保留高置信度选框
        //目标数据data:x,y,w,h为百分比，x,y为目标中心点坐标
            float* data = (float*)outs[i].data;
            for(int j=0;j<outs[i].rows;j++,data+=outs[i].cols){
                cv::Mat scores = outs[i].row(j).colRange(5,outs[i].cols);
                cv::Point classIdPoint;
                double confidence;//置信度
                //取得最大分数值与索引
                cv::minMaxLoc(scores,0,&confidence,0,&classIdPoint);
                if(confidence>confThreshold){
                    int centerX = (int)(data[0]*frame.cols);
                    int centerY = (int)(data[1]*frame.rows);
                    int width = (int)(data[2]*frame.cols);
                    int height = (int)(data[3]*frame.rows);
                    int left = centerX-width/2;
                    int top = centerY-height/2;
                    classIds.push_back(classIdPoint.x);
                           confidences.push_back((float)confidence);
                           boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
        //低置信度
        vector<int> indices;//保存没有重叠边框的索引
        //该函数用于抑制重叠边框
        cv::dnn::NMSBoxes(boxes,confidences,confThreshold,nmsThreshold,indices);
        for(size_t i=0;i<indices.size();i++){
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            drawPred(classIds[idx],confidences[idx],box.x,box.y,
            box.x+box.width,box.y+box.height,frame);
        }

}

void MainWindow::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat &frame)
{
    //绘制边界框
    cv::rectangle(frame,cv::Point(left,top),cv::Point(right,bottom),cv::Scalar(255,178,50),3);
    string label = cv::format("%.2f",conf);
    if(!classes.empty()){
        CV_Assert(classId < (int)classes.size());
        label = classes[classId]+":"+label;//边框上的类别标签与置信度
    }
    //绘制边界框上的标签
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label,cv::FONT_HERSHEY_SIMPLEX,0.5,1,&baseLine);
    top = max(top,labelSize.height);
    cv::rectangle(frame,cv::Point(left,top-round(1.5*labelSize.height)),cv::Point(left+round(1.5*labelSize.width),top+baseLine),cv::Scalar(255,255,255),cv::FILLED);
    cv::putText(frame, label,cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75,cv::Scalar(0, 0, 0), 1);

}
