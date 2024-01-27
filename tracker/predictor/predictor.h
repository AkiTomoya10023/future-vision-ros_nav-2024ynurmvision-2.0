#ifndef PREDICTOR_HPP
#define PREDICTOR_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "kalman.h"
#include "../general/general.h"
#include "../coordsolver/coordsolver.h"
#include "../serial/serial.h"

enum EnemyColor {RED, BLUE};
// 目标信息
struct TargetInfo
{
    Eigen::Vector3d xyz;
    int dist;
    int timestamp;
};

struct Armor
{
    int id;
    int color;
    int area;
    double conf;
    string key;
    Point2f apex2d[4];
    Rect rect;
    RotatedRect rrect;
    Rect roi;
    Point2f center2d;
    Eigen::Vector3d center3d_cam;
    Eigen::Vector3d center3d_world;
    Eigen::Vector3d euler;
    Eigen::Vector3d predict;
    std::vector<Point2f> points_pic_;
    float dis;

    TargetType type;
};

class ArmorPredictor
{

public:
    static constexpr int S = 2;

    EnemyColor enemy_color = RED;
#ifdef ENEMY_COLOR_BLUE
    enemy_color = BLUE;
#define ENEMY_COLOR_RED
    enemy_color = RED;
#endif
#define SHOW_IMG
#ifdef SHOW_IMG
    #define SHOW_ALL_ARMOR                        // 是否绘制装甲板
    #define SHOW_ALL_FANS                         // 是否绘制所有扇叶
    #define SHOW_FPS                              // 是否显示FPS
    #define SHOW_PREDICT                          // 是否显示预测
    #define SHOW_AIM_CROSS                        // 是否绘制十字瞄准线
#endif                                        // SHOW_IMG
    ArmorPredictor();
    ~ArmorPredictor();
    bool initParam();
    bool predict(Detection_pack &, MCUSend & , TaskData src);
    CoordSolver coordsolver;

private:
    Eigen::Matrix3d R_CI;           // 陀螺仪坐标系到相机坐标系旋转矩阵EIGEN-Matrix
    Eigen::Matrix3d F;             // 相机内参矩阵EIGEN-Matrix
    Eigen::Matrix<double, 1, 5> C; // 相机畸变矩阵EIGEN-Matrix
    cv::Mat R_CI_MAT;               // 陀螺仪坐标系到相机坐标系旋转矩阵CV-Mat
    cv::Mat F_MAT;                  // 相机内参矩阵CV-Mat
    cv::Mat C_MAT;                  // 相机畸变矩阵CV-Mat
    using _Kalman = Kalman<1, S>;
    _Kalman kalman;
    Armor last_armor;
    Armor final_armor;
    int dead_buffer_cnt;

    const int armor_type_wh_thres = 2.8;      //大小装甲板长宽比阈值

    const double armor_roi_expand_ratio_width = 1;
    const double armor_roi_expand_ratio_height = 2;

    const int max_lost_cnt = 5;                 //最大丢失目标帧数
    const size_t max_armors = 8;                   //视野中最多装甲板数
    const int max_dead_buffer = 2;              //允许因击打暂时熄灭的装甲板的出现次数
    const double max_delta_dist = 0.3;          //两次预测间最大速度(m/s)
    const double armor_conf_high_thres = 0.82;  //置信度大于该值的装甲板直接采用
    // const int max_delta_t = 50;              //使用同一预测器的最大时间间隔(ms)
    const size_t max_delta_t = 50;                //使用同一预测器的最大时间间隔(ms)
    const float max_IOU = 0.3;

};

class BuffPredictor
{
};

#endif // PREDICTOR_HPP
