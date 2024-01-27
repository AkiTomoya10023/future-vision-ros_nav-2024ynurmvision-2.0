#include "predictor.h"

using namespace yolox_detector;
ArmorPredictor::ArmorPredictor()
{
}

ArmorPredictor::~ArmorPredictor()
{
}

// 计算任意四边形的中心
cv::Point2f points_center(cv::Point2f pts[4])
{
    for (int i = 0; i < 4; ++i)
    {
        for (int j = i + 1; j < 4; ++j)
        {
            if (pts[i] == pts[j])
            {
                std::cout << "[Error] Unable to calculate center point." << std::endl;
                return cv::Point2f{0, 0};
            }
        }
    }
    cv::Point2f center(0, 0);
    if (pts[0].x == pts[2].x && pts[1].x == pts[3].x)
    {
        std::cout << "[Error] Unable to calculate center point." << std::endl;
    }
    else if (pts[0].x == pts[2].x && pts[1].x != pts[3].x)
    {
        center.x = pts[0].x;
        center.y = (pts[3].y - pts[1].y) / (pts[3].x - pts[1].x) * (pts[0].x - pts[3].x) + pts[3].y;
    }
    else if (pts[1].x == pts[3].x && pts[0].x != pts[2].x)
    {
        center.x = pts[1].x;
        center.y = (pts[2].y - pts[0].y) / (pts[2].x - pts[0].x) * (pts[1].x - pts[0].x) + pts[0].y;
    }
    else
    {
        center.x = (((pts[3].y - pts[1].y) / (pts[3].x - pts[1].x) * pts[3].x - pts[3].y +
                     pts[0].y - (pts[2].y - pts[0].y) / (pts[2].x - pts[0].x) * pts[0].x)) /
                   ((pts[3].y - pts[1].y) / (pts[3].x - pts[1].x) - (pts[2].y - pts[0].y) / (pts[2].x - pts[0].x));
        center.y = (pts[2].y - pts[0].y) / (pts[2].x - pts[0].x) * (center.x - pts[0].x) + pts[0].y;
    }

    return center;
}

static float getIOU(Armor cur_armor, Armor last_armor)
{
    // 左上坐标最大
    float xA = std::max(float(cur_armor.apex2d[0].x), last_armor.apex2d[0].x);
    float yA = std::max(cur_armor.apex2d[0].y, last_armor.apex2d[0].y);
    // 右下坐标最小
    float xB = std::min(cur_armor.apex2d[2].x, last_armor.apex2d[2].x);
    float yB = std::min(cur_armor.apex2d[2].y, last_armor.apex2d[2].y);

    float inter_area = std::max(0.0f, xB - xA + 1) * std::max(0.0f, yB - yA + 1);
    // 右下.x - 左上.x
    float boxAArea = (cur_armor.apex2d[2].x - cur_armor.apex2d[0].x + 1) * (cur_armor.apex2d[2].y - cur_armor.apex2d[0].y + 1);
    float boxBArea = (last_armor.apex2d[2].x - last_armor.apex2d[0].x + 1) * (last_armor.apex2d[2].y - last_armor.apex2d[0].y + 1);

    float IOU = inter_area / (boxAArea + boxBArea - inter_area);

    return IOU;
}

bool ArmorPredictor::initParam()
{
    // cv::FileStorage fin(param_path.c_str(), cv::FileStorage::READ);
    // fin["Tcb"] >> R_CI_MAT;
    // fin["K"] >> F_MAT;
    // fin["D"] >> C_MAT;
    // cv::cv2eigen(R_CI_MAT, R_CI);
    // cv::cv2eigen(F_MAT, F);
    // cv::cv2eigen(C_MAT, C);

    _Kalman::Matrix_xxd A = _Kalman::Matrix_xxd::Identity();
    _Kalman::Matrix_zxd H;
    H(0, 0) = 1;
    _Kalman::Matrix_xxd R;
    R(0, 0) = 0.01;
    for (int i = 1; i < S; i++)
    {
        R(i, i) = 100;
    }
    _Kalman::Matrix_zzd Q{4};
    _Kalman::Matrix_x1d init{0, 0};
    kalman = _Kalman(A, H, R, Q, init, 0);
}

bool ArmorPredictor::predict(Detection_pack &detect_pack, MCUSend &send_data, TaskData src)
{
    if (detect_pack.detection.empty())
    {
        return false;
    }
    // auto start = std::chrono::steady_clock::now();
    std::vector<Armor> armors;
    for (auto &object : detect_pack.detection)
    {
        Armor armor;
        armor.id = object.cls;
        armor.color = object.color;
        armor.conf = object.prob;
        armor.area = object.area;
#ifdef IGNORE_ENGINEER
        if (object.cls == 2)
            continue;
#endif // IGNORE_ENGINEER

#ifdef IGNORE_NPC
        if (object.cls == 0 || object.cls == 6 || object.cls == 7)
           continue;
#endif // IGNORE_NPC
	
        // 放行对应颜色装甲板或灰色装甲板 0：BLUE 1:RED 2:GREY 3:PURPLE
        if (enemy_color == RED)
            if (!(object.color == 1 || object.color == 2 || object.color == 3))
                continue;
        if (enemy_color == BLUE)
            if (!(object.color == 1 || object.color == 2 || object.color == 3))
                continue;

        // 如果装甲板为灰色且类别不为上次击打装甲板类别
        if (object.color == 2 && object.cls != last_armor.id)
            continue;
        // 如果为灰色但是>因击打暂时熄灭的装甲板的出现次数 (TODO:最大击打次数跟检测速度有关)
        if (object.color == 2 && object.cls == last_armor.id && dead_buffer_cnt >= max_dead_buffer)
            continue;

        // 生成Key
        if (object.color == 0)
            armor.key = "B" + to_string(object.cls);
        if (object.color == 1)
            armor.key = "R" + to_string(object.cls);
        if (object.color == 2)
            armor.key = "N" + to_string(object.cls);
        if (object.color == 3)
            armor.key = "P" + to_string(object.cls);

        // 生成顶点与装甲板二维中心点
        memcpy(armor.apex2d, object.apex, 4 * sizeof(cv::Point2f));
        // 生成装甲板旋转矩形和ROI
        std::vector<Point2f> points_pic(armor.apex2d, armor.apex2d + 4);
        armor.points_pic_ = points_pic;
        RotatedRect points_pic_rrect = minAreaRect(points_pic);
        armor.rrect = points_pic_rrect;
        auto bbox = points_pic_rrect.boundingRect();
        auto x = bbox.x - 0.5 * bbox.width * (armor_roi_expand_ratio_width - 1);
        auto y = bbox.y - 0.5 * bbox.height * (armor_roi_expand_ratio_height - 1);
        armor.roi = Rect(x,
                         y,
                         bbox.width * armor_roi_expand_ratio_width,
                         bbox.height * armor_roi_expand_ratio_height);

        // 计算长宽比,确定装甲板类型
        auto apex_wh_ratio = max(points_pic_rrect.size.height, points_pic_rrect.size.width) /
                             min(points_pic_rrect.size.height, points_pic_rrect.size.width);
        // 若大于长宽阈值或为哨兵、英雄装甲板
        // FIXME:若存在平衡步兵需要对此处步兵装甲板类型进行修改
        if (object.cls == 0 || object.cls == 1)
            armor.type = BIG;
        else if (object.cls == 2 || object.cls == 3 || object.cls == 4 || object.cls == 5 || object.cls == 6)
            armor.type = SMALL;
        else if (apex_wh_ratio > armor_type_wh_thres)
            armor.type = BIG;
        armors.push_back(armor);
    }
    if (armors.empty())
    {
        // 无目标，清空上次击打信息
        // last_armor.id = 0;
        return false;
    }

    std::sort(armors.begin(), armors.end(), [](Armor &prev, Armor &next)
              { return prev.area > next.area; });
    if (armors.size() > max_armors)
    {
        // 根据面积选择最大的8个目标
        armors.resize(max_armors);
    }
    int pnp_method;
    if (armors.size() <= 2)
        pnp_method = SOLVEPNP_ITERATIVE;
    else
        pnp_method = SOLVEPNP_IPPE;

#ifdef USING_IMU
    Eigen::Matrix3d rmat_imu = send_data.quat.toRotationMatrix();
    auto vec = rotationMatrixToEulerAngles(rmat_imu);
    // cout<<"Euler : "<<vec[0] * 180.f / CV_PI<<" "<<vec[1] * 180.f / CV_PI<<" "<<vec[2] * 180.f / CV_PI<<endl;
#else
    Eigen::Matrix3d rmat_imu = Eigen::Matrix3d::Identity();
#endif // USING_IMU

    std::vector<Armor> cur_armors;
    PnPInfo pnp_result;
    final_armor = armors.front();
    for (auto &armor_ : armors)
    {
        pnp_result = coordsolver.pnp(armor_.points_pic_, rmat_imu, armor_.type, pnp_method);
        if (pnp_result.armor_cam.norm() > 13 ||
            isnan(pnp_result.armor_cam[0]) ||
            isnan(pnp_result.armor_cam[1]) ||
            isnan(pnp_result.armor_cam[2]))
            continue;
        if (last_armor.id != 0)
        {
            float iou = getIOU(armor_, last_armor);
            if (iou > max_IOU)
            {
                final_armor = armor_;
                putText(detect_pack.img, fmt::format("{:.2f}", iou), final_armor.apex2d[1], FONT_HERSHEY_SIMPLEX, 1, {255, 255, 0}, 2);
            }
            // else{
            //     continue;
            // }
        }
    }
    if (final_armor.color == 2)
        dead_buffer_cnt++;
    else
        dead_buffer_cnt = 0;
    //        armor.type = target_type;
    final_armor.center3d_world = pnp_result.armor_world;
    final_armor.center3d_cam = pnp_result.armor_cam;
    final_armor.euler = pnp_result.euler;
    //        armor.area = object.area;
    //        armors.push_back(armor);
    auto angle = coordsolver.getAngle(final_armor.center3d_cam, rmat_imu);
    // 若预测出错则直接世界坐标系下坐标作为击打点
    if (isnan(angle[0]) || isnan(angle[1]))
        angle = coordsolver.getAngle(final_armor.center3d_cam, rmat_imu);
    last_armor = final_armor;

    static double last_yaw = 0, last_speed = 0;
    double yaw = angle[0];
    double pitch = angle[1];
    
    last_yaw = yaw;
    Eigen::Matrix<double, 1, 1> z_k{yaw};
    auto t = (double)(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - src.timestamp).count());
    std::cerr << "[ALL]: cost: " << std::to_string(t) << std::endl;
    _Kalman::Matrix_x1d state = kalman.update(z_k, t);
    last_speed = state(1, 0);
    double c_yaw = state(0, 0); // current yaw: yaw的滤波值，单位弧度
    if (isnan(c_yaw))
    {
        c_yaw = yaw;
        std::cerr << "[PRE]: Yaw solver ERROR, cost: " << std::to_string(t) << std::endl;
    }
    send_data = {0xAA, (float)c_yaw, (float)pitch, 0X55};

    // auto end = std::chrono::steady_clock::now();
    // std::cerr << "[PRE]: cost: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
#ifdef SHOW_ALL_ARMOR
    putText(detect_pack.img, fmt::format("Y:{:.2f} P:{:.2f}", send_data.yaw_angle, send_data.pitch_angle), final_armor.apex2d[2], FONT_HERSHEY_SIMPLEX, 1, {0, 255, 0}, 2);
    putText(detect_pack.img, fmt::format("{:.2f}", final_armor.conf), final_armor.apex2d[3], FONT_HERSHEY_SIMPLEX, 1, {0, 255, 0}, 2);
    if (final_armor.color == 0)
        putText(detect_pack.img, fmt::format("B{}", final_armor.id), final_armor.apex2d[0], FONT_HERSHEY_SIMPLEX, 1, {255, 100, 0}, 2);
    if (final_armor.color == 1)
        putText(detect_pack.img, fmt::format("R{}", final_armor.id), final_armor.apex2d[0], FONT_HERSHEY_SIMPLEX, 1, {0, 0, 255}, 2);
    if (final_armor.color == 2)
        putText(detect_pack.img, fmt::format("N{}", final_armor.id), final_armor.apex2d[0], FONT_HERSHEY_SIMPLEX, 1, {255, 255, 255}, 2);
    if (final_armor.color == 3)
        putText(detect_pack.img, fmt::format("P{}", final_armor.id), final_armor.apex2d[0], FONT_HERSHEY_SIMPLEX, 1, {255, 100, 255}, 2);
    for (int i = 0; i < 4; i++)
        line(detect_pack.img, final_armor.apex2d[i % 4], final_armor.apex2d[(i + 1) % 4], {0, 255, 0}, 1);
    rectangle(detect_pack.img, final_armor.roi, {255, 0, 255}, 1);
    auto armor_center = coordsolver.reproject(final_armor.center3d_cam);
    circle(detect_pack.img, armor_center, 4, {0, 0, 255}, 2);
#endif // SHOW_ALL_ARMOR
    return true;
}
