#include "thread.h"

/// Main function to run the whole program
GX_STATUS GxCamera::acquisitionStart(cv::Mat* targetMatImg)
{
    //////////////////////////////////////////SOME SETTINGS/////////////////////////////////////////////
    threadParam.m_hDevice = g_hDevice;
    threadParam.m_pImage = targetMatImg;
    threadParam.g_AcquisitionFlag = &g_bAcquisitionFlag;

    GX_STATUS emStatus;
    //Set Roi
    emStatus = setRoi();
    GX_VERIFY_EXIT(emStatus);
    //Set Exposure and Gain
    emStatus = setExposureGain();
    GX_VERIFY_EXIT(emStatus);
    //Set WhiteBalance
    emStatus = setWhiteBalance();
    GX_VERIFY_EXIT(emStatus);


    //Set acquisition mode
    emStatus = GXSetEnum(g_hDevice, GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
    GX_VERIFY_EXIT(emStatus);

    //Set trigger mode
    emStatus = GXSetEnum(g_hDevice, GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);
    GX_VERIFY_EXIT(emStatus);

    //Set buffer quantity of acquisition queue
    uint64_t nBufferNum = ACQ_BUFFER_NUM;
    emStatus = GXSetAcqusitionBufferNumber(g_hDevice, nBufferNum);
    GX_VERIFY_EXIT(emStatus);

    bool bStreamTransferSize = false;
    emStatus = GXIsImplemented(g_hDevice, GX_DS_INT_STREAM_TRANSFER_SIZE, &bStreamTransferSize);
    GX_VERIFY_EXIT(emStatus);

    if(bStreamTransferSize)
    {
        //Set size of data transfer block
        emStatus = GXSetInt(g_hDevice, GX_DS_INT_STREAM_TRANSFER_SIZE, ACQ_TRANSFER_SIZE);
        GX_VERIFY_EXIT(emStatus);
    }

    bool bStreamTransferNumberUrb = false;
    emStatus = GXIsImplemented(g_hDevice, GX_DS_INT_STREAM_TRANSFER_NUMBER_URB, &bStreamTransferNumberUrb);
    GX_VERIFY_EXIT(emStatus);

    if(bStreamTransferNumberUrb)
    {
        //Set qty. of data transfer block
        emStatus = GXSetInt(g_hDevice, GX_DS_INT_STREAM_TRANSFER_NUMBER_URB, ACQ_TRANSFER_NUMBER_URB);
        GX_VERIFY_EXIT(emStatus);
    }

    //Device start acquisition
    emStatus = GXStreamOn(g_hDevice);
    if(emStatus != GX_STATUS_SUCCESS)
    {
        GX_VERIFY_EXIT(emStatus);
    }

    //////////////////////////////////////////CREATE THREAD/////////////////////////////////////////////

    //Start acquisition thread, if thread create failed, exit this app
    int nRet = pthread_create(&g_nAcquisitonThreadID, NULL, ProcGetImage, (void*)&threadParam);
    if(nRet != 0)
    {
        GXCloseDevice(g_hDevice);
        g_hDevice = NULL;
        GXCloseLib();

        printf("<Failed to create the acquisition thread, App Exit!>\n");
        exit(nRet);
    }

    sleep(1);

    
    printf("????????????????loop is running???????????????????????\n");

    return 0;
}

GX_STATUS GxCamera::acquisitionStop()
{
    //////////////////////////////////////////STOP THREAD/////////////////////////////////////////////
    GX_STATUS emStatus;
    //Stop Acquisition thread
    g_bAcquisitionFlag = false;
    pthread_join(g_nAcquisitonThreadID, NULL);

    //Device stop acquisition
    emStatus = GXStreamOff(g_hDevice);
    if(emStatus != GX_STATUS_SUCCESS)
    {
        GX_VERIFY_EXIT(emStatus);
    }

    //Close device
    emStatus = GXCloseDevice(g_hDevice);
    if(emStatus != GX_STATUS_SUCCESS)
    {
        GetErrorString(emStatus);
        g_hDevice = NULL;
        GXCloseLib();
        exit(0);
    }

    //Release libary
    emStatus = GXCloseLib();
    if(emStatus != GX_STATUS_SUCCESS)
    {
        GetErrorString(emStatus);
        exit(0);
    }

    printf("<App exit!>\n");
    return 0;
}

/**
 * @brief 生产者线程
 * @param factory 工厂类
 **/
bool producer(Factory<TaskData> &factory)
{
    /* code */
    TaskData src;
#ifdef USING_VIDEO // Using video
    const string input_path = "../detector/assets/red_grey_car.avi";
    cv::VideoCapture video_cap(input_path);
    video_cap.open(input_path);
    assert(video_cap.isOpened());

    while (video_cap.read(src.img))
    {
        src.timestamp = std::chrono::steady_clock::now();
        factory.produce(src);
#ifdef SHOW_ALL_ARMOR
        cv::namedWindow("input", 0);
        cv::imshow("input", src.img);
        cv::waitKey(1);
#endif
        sleep(0.010);
    }
    video_cap.release();
#endif

#ifdef USING_DAHENG
    GxCamera camera;
    // init camrea lib
    camera.initLib();

    //   open device      SN号
    camera.openDevice("KE0200080468");

    // Attention:   (Width-64)%2=0; (Height-64)%2=0; X%16=0; Y%2=0;
    //    ROI             Width           Height       X       Y
    //  camera.setRoiParam(   640,            480,        320,     256);

    //   ExposureGain          autoExposure  autoGain  ExposureTime  AutoExposureMin  AutoExposureMax  Gain(<=16)  AutoGainMin  AutoGainMax  GrayValue
    camera.setExposureGainParam(true, false, 11000, 12000, 13000, 12, 5, 16, 127);

    //   WhiteBalance             Applied?       light source type
    camera.setWhiteBalanceParam(true, GX_AWB_LAMP_HOUSE_ADAPTIVE);

    //   Acquisition Start!
    camera.acquisitionStart(&(src.img));

    while (1)
    {
        src.timestamp = std::chrono::steady_clock::now();
#ifdef SHOW_ALL_ARMOR
        cv::namedWindow("input", 0);
        cv::imshow("input", src.img);
        cv::waitKey(1);
#endif
        factory.produce(src);
        sleep(0.010);
    }
    
    camera.acquisitionStop();
    
#endif
    return true;
}

/**
 * @brief 消费者线程
 * @param factory 工厂类
 **/
bool consumer(Factory<TaskData> &task_factory, Factory<MCUSend> &transmit_factory)
{
    // const string input_path = "../detector/assets/red_grey_car.avi";
    string engine_file_path = "../detector/model/tup_yolox3.engine";
    string param_path = "../params/coord_param.yaml";
    string camera_name = "KE0200080468";
    TaskData src;
    MCUSend send_data;
    Detection_pack detect_pack;
    yolox_detector::ArmorDetector detector;
    if (!(detector.initModel(engine_file_path)))
    {
        std::cerr << " Init model FAILED." << std::endl;
        return false;
    }

    ArmorPredictor predictor;
    predictor.coordsolver.loadParam(param_path, camera_name);
    predictor.initParam();
    while (1)
    {
        /* code */
        task_factory.consume(src);
        detect_pack.img = detector.detect(detect_pack.detection, src.img);
        if (predictor.predict(detect_pack, send_data, src))
        {
            transmit_factory.produce(send_data);
#ifdef SHOW_ALL_ARMOR
            cv::namedWindow("pre_rst", 0);
            cv::imshow("pre_rst", detect_pack.img);
            cv::waitKey(1);
#endif
        } 
    }
    return true;
}

/**
 * @brief 数据发送线程
 *
 * @param serial SerialPort类
 * @param transmit_factory Factory类
 * @return true
 * @return false
 */
bool dataTransmitter(Serial &serial, Factory<MCUSend> &transmit_factory)
{
    unsigned int cnt = 0;
    MCUSend send_data;
    while (1)
    {
        transmit_factory.consume(send_data);
        if (serial.send(send_data))
        {
            cnt++;
            if (cnt % 1000 == 0)
            {
                std::cout << "Send MCU data, Yaw: " << std::to_string(send_data.yaw_angle) << ", Pit: " << std::to_string(send_data.pitch_angle) << std::endl;
            }
        }
    }
    return true;
}

// /**
//  * @brief 串口监视线程
//  *
//  * @param serial
//  * @return true
//  * @return false
//  */
// bool serialWatcher(SerialPort &serial)
// {
//     int last = 0;
// #ifdef DEBUG_WITHOUT_COM
// #ifdef SAVE_TRANSMIT_LOG
//     LOG(WARNING) << "[SERIAL] Warning: You are not using Serial port";
// #endif // SAVE_TRANSMIT_LOG
// #endif // DEBUG_WITHOUT_COM

//     while (1)
//     {
//         sleep(1);
//         // 检测文件夹是否存在或串口需要初始化
//         if (access(serial.device.path.c_str(), F_OK) == -1 || serial.need_init)
//         {
//             serial.need_init = true;
// #ifdef DEBUG_WITHOUT_COM
//             int now = clock() / CLOCKS_PER_SEC;
//             if (now - last > 10)
//             {
//                 last = now;
//                 fmt::print(fmt::fg(fmt::color::orange), "[SERIAL] Warning: You are not using Serial port\n");
//             }
//             serial.withoutSerialPort();
// #else
//             serial.initSerialPort();
// #endif // DEBUG_WITHOUT_COM
//         }
//     }
// }

// #ifdef USING_IMU_WIT
// bool dataReceiver(IMUSerial &serial_imu, MessageFilter<MCUData> &receive_factory, std::chrono::_V2::steady_clock::time_point time_start)
// {
//     while(1)
//     {
//         //若串口离线则跳过数据发送
//         if (serial_imu.need_init == true)
//         {
//             // cout<<"offline..."<<endl;
//             continue;
//         }
//         if (!serial_imu.readData())
//         {
//             continue;
//         }
//         auto time_cap = std::chrono::steady_clock::now();
//         auto timestamp = (int)(std::chrono::duration<double,std::milli>(time_cap - time_start).count());
//         if (!serial_imu.is_quat_initialized)
//         {
//             continue;
//         }
//         Eigen::Quaterniond quat = serial_imu.quat;
//         Eigen::Vector3d acc = serial_imu.acc;
//         Eigen::Vector3d gyro =serial_imu.gyro;
//         MCUData imu_status = {acc, gyro, quat, timestamp};

//         receive_factory.produce(imu_status, timestamp);
//         Eigen::Vector3d vec = quat.toRotationMatrix().eulerAngles(2,1,0);
//     }
//     return true;
// }

// bool serialWatcher(SerialPort &serial, IMUSerial &serial_imu)
// {
//     while(1)
//     {
//         sleep(0.1);
//         //检测文件夹是否存在或串口需要初始化
//         if (access(serial.device.path.c_str(),F_OK) == -1 || serial.need_init)
//         {
//             serial.need_init = true;
//             serial.initSerialPort();
//         }
//         if (access(serial_imu.device.path.c_str(),F_OK) == -1 || serial_imu.need_init)
//         {
//             serial_imu.need_init = true;
//             serial_imu.initSerialPort();
//         }

//     }
// }
// #endif //USING_WIT_IMU

// #ifndef USING_IMU
// bool serialWatcher(SerialPort &serial)
// {
//     while(1)
//     {
//         sleep(0.1);
//         //检测文件夹是否存在或串口需要初始化
//         if (access(serial.device.path.c_str(),F_OK) == -1 || serial.need_init)
//         {
//             serial.need_init = true;
//             serial.initSerialPort();
//         }

//     }
// }
// #endif //USING_IMU
