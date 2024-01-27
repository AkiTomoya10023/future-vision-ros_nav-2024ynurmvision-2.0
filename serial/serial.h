#ifndef  _USART_H
#define  _USART_H
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdint.h>
#include <mutex>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <errno.h>
#include <string>
#include <string.h>
#include <stdexcept>


#include <exception>
#include <unistd.h>
#include <fstream>
#include <memory>
#include <memory.h>
#include <iomanip>

#undef EOF

struct MCUSend
{
//  //uint8_t name;
//  int id;
//  float delaytime;
//  //int distance;
//  float yaw_angle;
//  float pitch_angle;
//  //uint16_t detail;
    uint8_t sof;
    float yaw_angle;
    float pitch_angle;
    //float distance;
    uint8_t eof;
};

struct MCUReceive
{
//  int ctrl_mode;
//  int shoot_speed;
//  //int task_mode;
    char sof;
    char robot_id;
    char eof;

};


class Serial{

private:

    int _serialFd;

    enum
    {
        //JetsonCommSOF = (uint8_t)0x66,
        JetsonCommSOF = (uint8_t)0xAA,
        //JetsonCommEOF = (uint8_t)0x88
        JetsonCommEOF = (uint8_t)0x55
    };

public:

    int openPort(std::string portName);
    int closePort();
    int send(const MCUSend& test_send);
    MCUReceive receive();
    //Serial::Test_receive get_receive();
    MCUSend pack(const MCUSend& ctrl);
    //static uint16_t TransShort(uint8_t low, uint8_t high);
    //static float TransFloat(uint8_t f1, uint8_t f2, uint8_t f3, uint8_t f4);
};

#endif // Serial
