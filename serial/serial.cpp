#include "serial.h"

using namespace std;


MCUSend test_send_frame;
MCUReceive test_receive;

int Serial::openPort(std::string portName)
{
    _serialFd = open(portName.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (_serialFd == -1)
    {
        cout << "Open serial port failed." << endl;
        return -1;
    }

    termios tOption;                                // 串口配置结构体
    tcgetattr(_serialFd, &tOption);                 //获取当前设置
    cfmakeraw(&tOption);
    cfsetispeed(&tOption, B115200);                 // 接收波特率
    cfsetospeed(&tOption, B115200);                 // 发送波特率
    tcsetattr(_serialFd, TCSANOW, &tOption);
    tOption.c_cflag &= ~PARENB;
    tOption.c_cflag &= ~CSTOPB;
    tOption.c_cflag &= ~CSIZE;
    tOption.c_cflag |= CS8;
    tOption.c_cflag &= ~INPCK;
    tOption.c_cflag |= (B115200 | CLOCAL | CREAD);  // 设置波特率，本地连接，接收使能
    tOption.c_cflag &= ~(INLCR | ICRNL);
    tOption.c_cflag &= ~(IXON);
    tOption.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tOption.c_oflag &= ~OPOST;
    tOption.c_oflag &= ~(ONLCR | OCRNL);
    tOption.c_iflag &= ~(ICRNL | INLCR);
    tOption.c_iflag &= ~(IXON | IXOFF | IXANY);
    tOption.c_cc[VTIME] = 1;                        //只有设置为阻塞时这两个参数才有效
    tOption.c_cc[VMIN] = 1;
    tcflush(_serialFd, TCIOFLUSH);                  //TCIOFLUSH刷新输入、输出队列。

    //_test_receive = {1, 16};

    //cout << "Serial preparation complete." << endl;
    return 1;
}


int Serial::closePort()
{
    tcflush(_serialFd, TCIOFLUSH);
    if (-1 == close(_serialFd))
    {
        //cout << "Serial closing failed." << endl;
    }
    else
    {
        //cout << "Serial closing complete." << endl;
    }
    return 0;
}

MCUReceive Serial::receive()
{
    MCUReceive _t;

    memset(&_t, 0, sizeof(_t));
    int readCount = 0;
    if(readCount<int(sizeof(MCUReceive)))
    {
        int once = 0;

//    once = read(_serialFd, ((unsigned char *)(&_t)) + readCount, sizeof(Test_temp_receive) - readCount);
        once = read(_serialFd, &_t, sizeof(MCUReceive));
//    if (once == -1 || _t.sof != 0x55)
        if (once == -1)
        {
            printf("sof:%d, robot id:%d, eof:%d\n", _t.sof, _t.robot_id, _t.eof);
            cout<<"__________________****************recv failed!.....***********************"<<endl;
        }
        else
        {
            //cout<<"___________________****************recv success.....***********************"<<endl;
            //cout<<"ctrl_mode:  "<<_test_receive.ctrl_mode<<endl;
            //cout<<"shoot_speed:"<<_test_receive.shoot_speed<<endl;
//      _test_receive.robot_id = _t.robot_id;
            memcpy(&test_receive, &_t, sizeof(_t));
            printf("sof:%d, robot id:%d, eof:%d\n", test_receive.sof, test_receive.robot_id, test_receive.eof);
        }

        readCount += once;
    }
    tcflush(_serialFd, TCIFLUSH);

    //cout << hex << s.TransShort(_t.id3, _t.id4) << endl;

    //_test_receive.id = _t.SOF;
    //_test_receive.shoot_speed = Serial::TransFloat(_t.shoot_speed_1, _t.shoot_speed_2, _t.shoot_speed_3, _t.shoot_speed_4);
    //_test_receive.task_mode = _t.Test_task_mode;
    //cout<<"serial receive success..."<<endl;
    /*_test_receive.robot_level = _t.Test_robot_level;
    _test_receive.blood = _t.Test_blood;
    _test_receive.shoot_mode = _t.Test_shoot_mode;

    //_test_receive.detail = Serial::TransShort(_t.Test_uint16_1, _t.Test_uint16_2);

    _test_receive.power = Serial::TransFloat(_t.Test_power_1, _t.Test_power_2, _t.Test_power_3, _t.Test_power_4);y
    _test_receive.heat = Serial::TransFloat(_t.Test_heat_1, _t.Test_heat_2, _t.Test_heat_3, _t.Test_heat_4);
    _test_receive.yaw_real_angle = Serial::TransFloat(_t.yaw_real_angle_1, _t.yaw_real_angle_2, _t.yaw_real_angle_3, _t.yaw_real_angle_4);
    _test_receive.pitch_real_angle = Serial::TransFloat(_t.pitch_real_angle_1, _t.pitch_real_angle_2, _t.pitch_real_angle_3, _t.pitch_real_angle_4);*/

    return test_receive;
}

int Serial::send(const MCUSend& test_send)
{
    //_test_send_frame=pack(test_send);
    tcflush(_serialFd, TCOFLUSH);
    int sendCount;
    try
    {
        //sendCount  = write(_serialFd, &_test_send_frame, sizeof(Test_send_frame));
        sendCount  = write(_serialFd, &test_send, sizeof(MCUSend));
    }
    catch(exception e)
    {
        cout << e.what() << endl;
    }

    if (sendCount == -1)
    {

        cout << "\tSerial sending failed" << endl;
    }
    else if (sendCount < static_cast<int>(sizeof(MCUSend)))
    {

        cout << "\tSerial sending failed. "<< sizeof(MCUSend) - sendCount <<" bytes unsent." << endl;

    }
    else
    {

        cout << "\tSerial sending succeeded. " << endl;

    }

    return 1;
}

//Serial::Test_send_frame Serial::pack(const Test_send& ctrl)
//{
//    return Test_send_frame
//    {
//        JetsonCommSOF,
//        //ctrl.name,
//        //ctrl.id,
//        ctrl.id,
//        ctrl.delaytime,
//        //ctrl.distance,
//        ctrl.yaw_angle,
//        ctrl.pitch_angle,

//        //ctrl.detail,
//        JetsonCommEOF
//    };
//}

//Serial::Test_receive Serial::get_receive()
//{
//	return _test_receive;
//}

//uint16_t Serial::TransShort(uint8_t low, uint8_t high)
//{
//  uint16_t result;
//  uint16_t Testlow, Testhigh;
//  Testlow = (uint16_t)low;
//  Testhigh = ((uint16_t)high)<<8;
//  result = Testhigh + Testlow;
//  return result;
//}

//float Serial::TransFloat(uint8_t f1, uint8_t f2, uint8_t f3, uint8_t f4)
//{
//  int32_t tem;
//  int32_t* flo_rec;
//  float result;
//  tem = ((int32_t)f1) + (((int32_t)f2) << 8) + (((int32_t)f3) << 16) + (((int32_t)f4) << 24);
//  flo_rec = &tem;
//  result = *((float*)flo_rec);
//  return result;
//}


