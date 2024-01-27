#include <opencv2/opencv.hpp>
#include "./detector/inference.h"
#include "./tracker/coordsolver/coordsolver.h"
#include "./tracker/predictor/kalman.h"
#include "./tracker/predictor/predictor.h"
#include "debug.h"
#include "./tracker/general/general.h"
#include "./serial/serial.h"
#include "./thread/thread.h"

const string PORT_NAME = "/dev/ttyTHS0";

const int BAUD = 115200;
// const int BAUD_IMU = 460800;

int main(int argc, char *argv[])
{
    Factory<TaskData> task_factory(3);
    Factory<MCUSend> data_transmit_factory(5);

    // MessageFilter<MCUData> data_receiver(100);
    Serial serial;
    serial.openPort(PORT_NAME);

    std::thread task_producer(producer, std::ref(task_factory));

    // std::thread task_consumer(consumer, std::ref(task_factory), std::ref(task2));  // 编译通过
    std::thread task_consumer(consumer, std::ref(task_factory), std::ref(data_transmit_factory));  // error: std::thread arguments must be invocable after conversion to rvalues


    std::thread transmitter(dataTransmitter, ref(serial), std::ref(data_transmit_factory));

    task_producer.join();
    task_consumer.join();
    transmitter.join();

}
