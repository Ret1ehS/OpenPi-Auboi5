#include "aubo_sdk/rpc.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <limits>
#include <pthread.h>
#include <sched.h>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <poll.h>
#include <unistd.h>

using namespace arcs::aubo_sdk;
using namespace arcs::common_interface;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDefaultMoveLineAcc = 1.2;
constexpr double kDefaultMoveLineSpeed = 0.10;
constexpr double kDefaultMoveLineBlendRadius = 0.01;
constexpr double kMoveLineCheckSampleDist = 0.0025;

using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;
using Mat4 = std::array<std::array<double, 4>, 4>;

int choose_servo_scheduler_core()
{
    const unsigned int cpu_count = std::max(1u, std::thread::hardware_concurrency());
    return static_cast<int>(cpu_count - 1);
}

bool pin_current_thread_to_core(int core_id, int &error_code)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    const int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    error_code = ret;
    return ret == 0;
}

bool set_current_thread_fifo_max_priority(int &priority, int &error_code)
{
    errno = 0;
    priority = sched_get_priority_max(SCHED_FIFO);
    if (priority == -1) {
        error_code = errno;
        return false;
    }
    sched_param param{};
    param.sched_priority = priority;
    const int ret = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    error_code = ret;
    return ret == 0;
}

void configure_servo_thread_scheduling()
{
    const int core_id = choose_servo_scheduler_core();
    int affinity_err = 0;
    const bool affinity_ok = pin_current_thread_to_core(core_id, affinity_err);
    std::cout << "servo_thread_core=" << core_id << std::endl;
    std::cout << "servo_thread_affinity_ok=" << affinity_ok << std::endl;
    if (!affinity_ok) {
        std::cout << "servo_thread_affinity_err=" << affinity_err << std::endl;
    }

    int priority = 0;
    int priority_err = 0;
    const bool priority_ok = set_current_thread_fifo_max_priority(priority, priority_err);
    std::cout << "servo_thread_priority_policy=SCHED_FIFO" << std::endl;
    std::cout << "servo_thread_priority=" << priority << std::endl;
    std::cout << "servo_thread_priority_ok=" << priority_ok << std::endl;
    if (!priority_ok) {
        std::cout << "servo_thread_priority_err=" << priority_err << std::endl;
        std::cout << "servo_thread_priority_errstr=" << std::strerror(priority_err) << std::endl;
    }
}

struct Options {
    std::string robot_ip = "192.168.1.100";
    int port = 30004;
    std::string user = "aubo";
    std::string password = "123456";
    double speed_deg = 10.0;
    double acc_deg = 20.0;
    double speed_fraction = 1.0;
    bool execute = false;
    bool has_target_pose = false;
    bool has_joint_target = false;
    bool has_track_pose_file = false;
    bool daemon = false;
    std::vector<double> target_pose;
    std::vector<double> joint_target;
    std::string track_pose_file;
    double track_time_s = 0.0125;
    double smooth_scale = 0.5;
    double delay_scale = 1.0;
};

void print_vec(const char *name, const std::vector<double> &values)
{
    std::cout << name << "=[";
    for (size_t i = 0; i < values.size(); ++i) {
        std::cout << values[i];
        if (i + 1 < values.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

Mat4 eye4()
{
    return Mat4{ { { 1.0, 0.0, 0.0, 0.0 },
                    { 0.0, 1.0, 0.0, 0.0 },
                    { 0.0, 0.0, 1.0, 0.0 },
                    { 0.0, 0.0, 0.0, 1.0 } } };
}

Mat4 matmul(const Mat4 &a, const Mat4 &b)
{
    Mat4 out = {};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) {
                sum += a[i][k] * b[k][j];
            }
            out[i][j] = sum;
        }
    }
    return out;
}

Mat4 make_transform(const Vec3 &pos, const Quat &quat_wxyz)
{
    double w = quat_wxyz[0];
    double x = quat_wxyz[1];
    double y = quat_wxyz[2];
    double z = quat_wxyz[3];
    const double norm = std::sqrt(w * w + x * x + y * y + z * z);
    if (norm > 1e-12) {
        w /= norm;
        x /= norm;
        y /= norm;
        z /= norm;
    } else {
        w = 1.0;
        x = 0.0;
        y = 0.0;
        z = 0.0;
    }

    Mat4 t = eye4();
    t[0][0] = 1.0 - 2.0 * (y * y + z * z);
    t[0][1] = 2.0 * (x * y - z * w);
    t[0][2] = 2.0 * (x * z + y * w);
    t[1][0] = 2.0 * (x * y + z * w);
    t[1][1] = 1.0 - 2.0 * (x * x + z * z);
    t[1][2] = 2.0 * (y * z - x * w);
    t[2][0] = 2.0 * (x * z - y * w);
    t[2][1] = 2.0 * (y * z + x * w);
    t[2][2] = 1.0 - 2.0 * (x * x + y * y);
    t[0][3] = pos[0];
    t[1][3] = pos[1];
    t[2][3] = pos[2];
    return t;
}

Mat4 rotz(double theta)
{
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    Mat4 t = eye4();
    t[0][0] = c;
    t[0][1] = -s;
    t[1][0] = s;
    t[1][1] = c;
    return t;
}

std::map<std::string, Vec3> compute_positions(const std::vector<double> &q, const std::vector<double> &tcp_pose)
{
    static const std::vector<std::tuple<std::string, Vec3, Quat>> chain = {
        { "shoulder_Link", { 0.0, 0.0, 0.122 }, { 0.0, 0.0, 0.0, 1.0 } },
        { "upperArm_Link", { 0.0, 0.1215, 0.0 }, { 0.5, -0.5, -0.5, -0.5 } },
        { "foreArm_Link", { 0.408, 0.0, 0.0 }, { 0.0, -1.0, 0.0, 0.0 } },
        { "wrist1_Link", { 0.376, 0.0, 0.0 }, { 0.0, 0.707107, 0.707107, 0.0 } },
        { "wrist2_Link", { 0.0, 0.1025, 0.0 }, { 0.707107, -0.707107, 0.0, 0.0 } },
        { "wrist3_Link", { 0.0, -0.094, 0.0 }, { 0.707107, 0.707107, 0.0, 0.0 } },
    };

    Mat4 t = eye4();
    std::map<std::string, Vec3> positions;
    positions["base_link"] = { 0.0, 0.0, 0.0 };

    for (size_t idx = 0; idx < chain.size() && idx < q.size(); ++idx) {
        const auto &[name, pos, quat] = chain[idx];
        t = matmul(matmul(t, make_transform(pos, quat)), rotz(q[idx]));
        positions[name] = { t[0][3], t[1][3], t[2][3] };
    }

    if (tcp_pose.size() >= 3) {
        positions["tcp"] = { tcp_pose[0], tcp_pose[1], tcp_pose[2] };
    } else if (positions.count("wrist3_Link") > 0) {
        positions["tcp"] = positions["wrist3_Link"];
    } else {
        positions["tcp"] = { 0.0, 0.0, 0.0 };
    }
    return positions;
}

std::vector<std::vector<double>> read_pose_file(const std::string &path)
{
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("failed to open pose file: " + path);
    }

    std::vector<std::vector<double>> poses;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        std::vector<double> pose;
        double value = 0.0;
        while (iss >> value) {
            pose.push_back(value);
        }
        if (!pose.empty()) {
            if (pose.size() != 6) {
                throw std::runtime_error("each pose row must contain 6 floats");
            }
            poses.push_back(pose);
        }
    }
    return poses;
}

bool parse_args(int argc, char **argv, Options &opt)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const char *flag) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << flag << std::endl;
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--robot-ip") {
            opt.robot_ip = require_value("--robot-ip");
        } else if (arg == "--port") {
            opt.port = std::stoi(require_value("--port"));
        } else if (arg == "--user") {
            opt.user = require_value("--user");
        } else if (arg == "--password") {
            opt.password = require_value("--password");
        } else if (arg == "--speed-deg") {
            opt.speed_deg = std::stod(require_value("--speed-deg"));
        } else if (arg == "--acc-deg") {
            opt.acc_deg = std::stod(require_value("--acc-deg"));
        } else if (arg == "--speed-fraction") {
            opt.speed_fraction = std::stod(require_value("--speed-fraction"));
        } else if (arg == "--execute") {
            opt.execute = true;
        } else if (arg == "--daemon") {
            opt.daemon = true;
        } else if (arg == "--target-pose") {
            opt.has_target_pose = true;
            opt.target_pose.clear();
            for (int j = 0; j < 6; ++j) {
                opt.target_pose.push_back(std::stod(require_value("--target-pose")));
            }
        } else if (arg == "--joint-target") {
            opt.has_joint_target = true;
            opt.joint_target.clear();
            for (int j = 0; j < 6; ++j) {
                opt.joint_target.push_back(std::stod(require_value("--joint-target")));
            }
        } else if (arg == "--track-pose-file") {
            opt.has_track_pose_file = true;
            opt.track_pose_file = require_value("--track-pose-file");
        } else if (arg == "--track-time-s") {
            opt.track_time_s = std::stod(require_value("--track-time-s"));
        } else if (arg == "--smooth-scale") {
            opt.smooth_scale = std::stod(require_value("--smooth-scale"));
        } else if (arg == "--delay-scale") {
            opt.delay_scale = std::stod(require_value("--delay-scale"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: tcp_control_helper [--target-pose x y z rx ry rz | --joint-target q1..q6]\n"
                << "                          [--track-pose-file path --track-time-s 0.05]\n"
                << "                          [--execute] [--daemon] [--robot-ip IP] [--port 30004]\n";
            return false;
        } else {
            std::cerr << "unknown argument: " << arg << std::endl;
            return false;
        }
    }
    return true;
}

int wait_for_robot_mode(RobotInterfacePtr robot, RobotModeType target, int timeout_ms)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (robot->getRobotState()->getRobotModeType() == target) {
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return -1;
}

int wait_arrival_with_safety(RobotInterfacePtr robot, double stop_time_s)
{
    int retry = 0;
    int exec_id = robot->getMotionControl()->getExecId();
    while (exec_id == -1 && retry++ < 20) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        exec_id = robot->getMotionControl()->getExecId();
    }
    if (exec_id == -1) {
        return -1;
    }
    while (robot->getMotionControl()->getExecId() != -1) {
        auto state = robot->getRobotState();
        if (state->isCollisionOccurred() || !state->isWithinSafetyLimits()) {
            robot->getMotionControl()->stopJoint(stop_time_s);
            return -2;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}

int wait_for_exec_start(RobotInterfacePtr robot, int timeout_ms)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (robot->getMotionControl()->getExecId() != -1) {
            return robot->getMotionControl()->getExecId();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return -1;
}

bool safety_mode_allows_motion(SafetyModeType mode)
{
    return mode == SafetyModeType::Normal || mode == SafetyModeType::ReducedMode ||
           mode == SafetyModeType::Recovery;
}

bool safety_mode_needs_board_restart(SafetyModeType mode)
{
    return mode == SafetyModeType::Fault || mode == SafetyModeType::RobotEmergencyStop ||
           mode == SafetyModeType::SystemEmergencyStop;
}

bool robot_state_motion_ready(RobotStatePtr state)
{
    const auto safety_mode = state->getSafetyModeType();
    const bool within_limits = state->isWithinSafetyLimits() || safety_mode == SafetyModeType::Recovery;
    return !state->isCollisionOccurred() && within_limits && safety_mode_allows_motion(safety_mode);
}

int wait_for_motion_idle(RobotInterfacePtr robot, MotionControlPtr motion, int timeout_ms)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        auto state = robot->getRobotState();
        if (motion->getExecId() == -1 && state->isSteady()) {
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return -1;
}

int clear_abnormal_state_sdk(RobotInterfacePtr robot, MotionControlPtr motion, RobotManagePtr manage)
{
    auto state = robot->getRobotState();
    if (robot_state_motion_ready(state)) {
        return 0;
    }

    std::cout << "clear_state_before_safety_mode=" << state->getSafetyModeType() << std::endl;
    std::cout << "clear_state_before_collision=" << state->isCollisionOccurred() << std::endl;
    std::cout << "clear_state_before_within_limits=" << state->isWithinSafetyLimits() << std::endl;

    if (motion->getExecId() != -1 || !state->isSteady()) {
        const int stop_ret = motion->stopMove(true, true);
        const int clear_path_ret = motion->clearPath();
        std::cout << "clear_state_stop_move_ret=" << stop_ret << std::endl;
        std::cout << "clear_state_clear_path_ret=" << clear_path_ret << std::endl;
        std::cout << "clear_state_wait_idle_ret=" << wait_for_motion_idle(robot, motion, 1000) << std::endl;
    }

    state = robot->getRobotState();
    const auto safety_mode = state->getSafetyModeType();
    if (state->isCollisionOccurred() || safety_mode == SafetyModeType::ProtectiveStop) {
        const int unlock_ret = manage->setUnlockProtectiveStop();
        std::cout << "unlock_protective_stop_ret=" << unlock_ret << std::endl;
    }

    state = robot->getRobotState();
    if (safety_mode_needs_board_restart(state->getSafetyModeType())) {
        const int restart_ret = manage->restartInterfaceBoard();
        std::cout << "restart_interface_board_ret=" << restart_ret << std::endl;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(3000);
    while (std::chrono::steady_clock::now() < deadline) {
        state = robot->getRobotState();
        if (robot_state_motion_ready(state)) {
            std::cout << "clear_state_after_safety_mode=" << state->getSafetyModeType() << std::endl;
            std::cout << "clear_state_after_collision=" << state->isCollisionOccurred() << std::endl;
            std::cout << "clear_state_after_within_limits=" << state->isWithinSafetyLimits() << std::endl;
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    state = robot->getRobotState();
    std::cout << "clear_state_after_safety_mode=" << state->getSafetyModeType() << std::endl;
    std::cout << "clear_state_after_collision=" << state->isCollisionOccurred() << std::endl;
    std::cout << "clear_state_after_within_limits=" << state->isWithinSafetyLimits() << std::endl;
    return -1;
}

int ensure_robot_ready_for_motion(RobotInterfacePtr robot, MotionControlPtr motion, RobotManagePtr manage)
{
    const int clear_ret = clear_abnormal_state_sdk(robot, motion, manage);
    std::cout << "clear_abnormal_ret=" << clear_ret << std::endl;

    auto state = robot->getRobotState();
    if (!state->isPowerOn()) {
        const int pret = manage->poweron();
        std::cout << "poweron_ret=" << pret << std::endl;
        if (wait_for_robot_mode(robot, RobotModeType::Idle, 10000) != 0 && !state->isPowerOn()) {
            return -1;
        }
        state = robot->getRobotState();
    }

    if (state->getRobotModeType() != RobotModeType::Running) {
        const int sret = manage->startup();
        std::cout << "startup_ret=" << sret << std::endl;
        if (wait_for_robot_mode(robot, RobotModeType::Running, 15000) != 0) {
            return -2;
        }
        state = robot->getRobotState();
    }

    return robot_state_motion_ready(state) ? 0 : -3;
}

const char *servo_ret_name(int code);

int run_track_pose_sequence(RobotInterfacePtr robot,
                            MotionControlPtr motion,
                            RobotAlgorithmPtr algo,
                            RobotConfigPtr config,
                            const Options &opt)
{
    const auto poses = read_pose_file(opt.track_pose_file);
    std::cout << "track_pose_file=" << opt.track_pose_file << std::endl;
    std::cout << "track_steps=" << poses.size() << std::endl;
    std::cout << "track_time_s=" << opt.track_time_s << std::endl;
    std::cout << "track_smooth_scale=" << opt.smooth_scale << std::endl;
    std::cout << "track_delay_scale=" << opt.delay_scale << std::endl;
    std::cout << "speed_fraction=" << opt.speed_fraction << std::endl;
    std::cout << "exec_mode=servoCartesian" << std::endl;

    if (poses.empty()) {
        std::cerr << "track pose file is empty" << std::endl;
        return 27;
    }
    if (!opt.execute) {
        return 0;
    }

    auto state = robot->getRobotState();
    if (state->isCollisionOccurred() || !state->isWithinSafetyLimits()) {
        std::cerr << "track execute aborted because robot is already in collision or outside safety limits" << std::endl;
        return 28;
    }

    motion->setSpeedFraction(opt.speed_fraction);
    const int servo_mode_before = motion->getServoModeSelect();
    const double cycle_time_s = config->getCycletime();
    const double lookahead_time = std::max(0.03, std::min(0.2, cycle_time_s));
    const double gain = 150.0;
    std::cout << "servo_mode_before=" << servo_mode_before << std::endl;
    std::cout << "servo_cycle_time_s=" << cycle_time_s << std::endl;
    std::cout << "servo_lookahead_time_s=" << lookahead_time << std::endl;
    std::cout << "servo_gain=" << gain << std::endl;
    const int servo_enter_ret = motion->setServoModeSelect(1);
    const int servo_mode_after_enter = motion->getServoModeSelect();
    std::cout << "servo_enter_ret=" << servo_enter_ret << std::endl;
    std::cout << "servo_mode_after_enter=" << servo_mode_after_enter << std::endl;
    if (servo_enter_ret != 0 || servo_mode_after_enter != 1) {
        return 33;
    }

    auto cleanup_servo = [&]() {
        const int servo_exit_ret = motion->setServoModeSelect(0);
        std::cout << "servo_exit_ret=" << servo_exit_ret << std::endl;
        std::cout << "servo_mode_after_exit=" << motion->getServoModeSelect() << std::endl;
    };

    std::vector<double> seed_q = state->getJointPositions();
    int sent = 0;
    int last_servo_ret = 0;

    for (size_t idx = 0; idx < poses.size(); ++idx) {
        std::vector<double> pose = poses[idx];

        auto ik_result = algo->inverseKinematics(seed_q, pose);
        std::vector<double> target_q = std::get<0>(ik_result);
        const int ik_ret = std::get<1>(ik_result);
        if (ik_ret != 0 || target_q.size() != 6) {
            std::cout << "track_fail_index=" << idx << std::endl;
            std::cout << "track_ik_ret=" << ik_ret << std::endl;
            cleanup_servo();
            return 29;
        }

        last_servo_ret = 0;
        for (int retry = 0; retry < 5; ++retry) {
            last_servo_ret = motion->servoCartesian(pose, 0.0, 0.0, opt.track_time_s, lookahead_time, gain);
            if (last_servo_ret == 0) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (last_servo_ret != 0) {
            std::cout << "track_fail_index=" << idx << std::endl;
            std::cout << "servo_ret=" << last_servo_ret << std::endl;
            std::cout << "servo_ret_name=" << servo_ret_name(last_servo_ret) << std::endl;
            cleanup_servo();
            return 31;
        }

        seed_q = target_q;
        ++sent;
        std::this_thread::sleep_for(std::chrono::duration<double>(opt.track_time_s));

        state = robot->getRobotState();
        if (state->isCollisionOccurred() || !state->isWithinSafetyLimits()) {
            motion->stopJoint(1.0);
            cleanup_servo();
            std::cout << "track_fail_index=" << idx << std::endl;
            std::cout << "track_abort_reason=safety" << std::endl;
            std::cout << "servo_sent=" << sent << std::endl;
            return 32;
        }
    }

    std::this_thread::sleep_for(std::chrono::duration<double>(opt.track_time_s));
    cleanup_servo();
    const auto final_q = robot->getRobotState()->getJointPositions();
    print_vec("final_q_rad", final_q);
    std::cout << "servo_sent=" << sent << std::endl;
    std::cout << "servo_ret=" << last_servo_ret << std::endl;
    std::cout << "collision_after=" << robot->getRobotState()->isCollisionOccurred() << std::endl;
    std::cout << "within_safety_limits_after=" << robot->getRobotState()->isWithinSafetyLimits() << std::endl;
    std::cout << "robot_mode_after=" << robot->getRobotState()->getRobotModeType() << std::endl;
    std::cout << "safety_mode_after=" << robot->getRobotState()->getSafetyModeType() << std::endl;
    return 0;
}

// ---------------------------------------------------------------------------
// Daemon mode: keep connection alive, read commands from stdin
// ---------------------------------------------------------------------------
void print_snapshot(RobotInterfacePtr robot, RobotConfigPtr config)
{
    auto state = robot->getRobotState();
    auto joint_q = state->getJointPositions();
    auto tcp_pose = state->getTcpPose();
    auto tool_pose = state->getToolPose();
    auto tcp_offset = state->getActualTcpOffset();
    auto elbow_pos = state->getElbowPosistion();

    std::cout << "robot_mode=" << state->getRobotModeType() << std::endl;
    std::cout << "safety_mode=" << state->getSafetyModeType() << std::endl;
    std::cout << "is_power_on=" << state->isPowerOn() << std::endl;
    std::cout << "collision=" << state->isCollisionOccurred() << std::endl;
    std::cout << "within_safety_limits=" << state->isWithinSafetyLimits() << std::endl;
    std::cout << "collision_level=" << config->getCollisionLevel() << std::endl;
    std::cout << "collision_stop_type=" << config->getCollisionStopType() << std::endl;
    print_vec("joint_q_rad", joint_q);
    print_vec("tcp_pose", tcp_pose);
    print_vec("tool_pose", tool_pose);
    print_vec("tcp_offset", tcp_offset);
    print_vec("elbow_pos", elbow_pos);
}

// Parse a line of 6 space-separated doubles into a vector.
bool parse_pose6(const std::string &line, std::vector<double> &out)
{
    out.clear();
    std::istringstream iss(line);
    double v;
    while (iss >> v) {
        out.push_back(v);
    }
    return out.size() == 6;
}

double pose_linear_distance(const std::vector<double> &a, const std::vector<double> &b)
{
    if (a.size() < 3 || b.size() < 3) {
        return 0.0;
    }
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double compute_move_line_blend_radius(const std::vector<std::vector<double>> &poses,
                                      size_t idx,
                                      double requested_radius)
{
    if (requested_radius <= 0.0 || poses.size() < 2 || idx + 1 >= poses.size()) {
        return 0.0;
    }
    const double next_dist = pose_linear_distance(poses[idx], poses[idx + 1]);
    double limit_dist = next_dist;
    if (idx > 0) {
        limit_dist = std::min(limit_dist, pose_linear_distance(poses[idx - 1], poses[idx]));
    }
    const double radius = std::min(requested_radius, 0.35 * limit_dist);
    return radius >= 1e-3 ? radius : 0.0;
}

double wrap_angle(double angle)
{
    return std::atan2(std::sin(angle), std::cos(angle));
}

std::vector<double> interpolate_pose6(const std::vector<double> &a,
                                      const std::vector<double> &b,
                                      double t)
{
    std::vector<double> out(6, 0.0);
    for (size_t i = 0; i < 3 && i < a.size() && i < b.size(); ++i) {
        out[i] = a[i] + (b[i] - a[i]) * t;
    }
    for (size_t i = 3; i < 6 && i < a.size() && i < b.size(); ++i) {
        out[i] = a[i] + wrap_angle(b[i] - a[i]) * t;
    }
    return out;
}

bool validate_pose_sample(RobotAlgorithmPtr algo,
                          const std::vector<double> &pose,
                          std::vector<double> &seed_q,
                          std::string &reason,
                          int &ik_ret)
{
    auto ik_result = algo->inverseKinematics(seed_q, pose);
    auto target_q = std::get<0>(ik_result);
    ik_ret = std::get<1>(ik_result);
    if (ik_ret != 0 || target_q.size() != 6) {
        if (ik_ret == 0) {
            ik_ret = -1;
        }
        std::ostringstream oss;
        oss << "ik_ret=" << ik_ret;
        reason = oss.str();
        return false;
    }
    seed_q = target_q;
    return true;
}

bool validate_linear_segment_samples(RobotAlgorithmPtr algo,
                                     const std::vector<double> &start_pose,
                                     const std::vector<double> &end_pose,
                                     const std::vector<double> &start_q,
                                     size_t segment_index,
                                     std::string &error_kind,
                                     std::string &reason,
                                     int &ik_ret,
                                     size_t &sample_index)
{
    const double dist = pose_linear_distance(start_pose, end_pose);
    const int sample_count = std::max(1, static_cast<int>(std::ceil(dist / kMoveLineCheckSampleDist)));
    std::vector<double> seed_q = start_q;
    for (int sample = 1; sample < sample_count; ++sample) {
        const double t = static_cast<double>(sample) / static_cast<double>(sample_count);
        const auto pose = interpolate_pose6(start_pose, end_pose, t);
        if (!validate_pose_sample(
                algo,
                pose,
                seed_q,
                reason,
                ik_ret)) {
            error_kind = "segment_ik_fail";
            sample_index = static_cast<size_t>(sample);
            return false;
        }
    }
    return true;
}

bool validate_blend_segment_samples(RobotAlgorithmPtr algo,
                                    const std::vector<std::vector<double>> &poses,
                                    const std::vector<std::vector<double>> &joint_solutions,
                                    size_t via_index,
                                    double blend_radius,
                                    std::string &error_kind,
                                    std::string &reason,
                                    int &ik_ret,
                                    size_t &sample_index)
{
    if (via_index == 0 || via_index + 1 >= poses.size() || via_index >= joint_solutions.size()) {
        return true;
    }
    if (blend_radius <= 0.0) {
        return true;
    }

    std::vector<std::vector<double>> blend_points;
    try {
        blend_points = algo->pathBlend3Points(
            3,
            joint_solutions[via_index - 1],
            joint_solutions[via_index],
            joint_solutions[via_index + 1],
            blend_radius,
            kMoveLineCheckSampleDist
        );
    } catch (const std::exception &exc) {
        error_kind = "blend_path_exception";
        reason = std::string("pathBlend3Points failed: ") + exc.what();
        ik_ret = 0;
        sample_index = 0;
        return false;
    }

    std::vector<double> seed_q = joint_solutions[via_index - 1];
    for (size_t sample = 0; sample < blend_points.size(); ++sample) {
        if (blend_points[sample].size() < 3) {
            continue;
        }
        std::vector<double> pose = poses[via_index];
        pose[0] = blend_points[sample][0];
        pose[1] = blend_points[sample][1];
        pose[2] = blend_points[sample][2];
        if (!blend_points.empty()) {
            const double t = static_cast<double>(sample + 1) / static_cast<double>(blend_points.size() + 1);
            if (t <= 0.5) {
                const auto orient_pose = interpolate_pose6(poses[via_index - 1], poses[via_index], t * 2.0);
                pose[3] = orient_pose[3];
                pose[4] = orient_pose[4];
                pose[5] = orient_pose[5];
            } else {
                const auto orient_pose = interpolate_pose6(poses[via_index], poses[via_index + 1], (t - 0.5) * 2.0);
                pose[3] = orient_pose[3];
                pose[4] = orient_pose[4];
                pose[5] = orient_pose[5];
            }
        }
        if (!validate_pose_sample(
                algo,
                pose,
                seed_q,
                reason,
                ik_ret)) {
            error_kind = "blend_ik_fail";
            sample_index = sample;
            return false;
        }
    }
    return true;
}

bool validate_movel_chunk_swept_path(RobotAlgorithmPtr algo,
                                     const std::vector<std::vector<double>> &poses,
                                     const std::vector<std::vector<double>> &joint_solutions,
                                     double requested_blend_radius,
                                     std::string &error_kind,
                                     std::string &reason,
                                     int &ik_ret,
                                     size_t &fail_index,
                                     size_t &sample_index)
{
    if (poses.size() < 2 || joint_solutions.size() != poses.size()) {
        return true;
    }

    for (size_t idx = 0; idx + 1 < poses.size(); ++idx) {
        if (!validate_linear_segment_samples(
                algo,
                poses[idx],
                poses[idx + 1],
                joint_solutions[idx],
                idx,
                error_kind,
                reason,
                ik_ret,
                sample_index)) {
            fail_index = idx;
            return false;
        }
    }

    for (size_t idx = 1; idx + 1 < poses.size(); ++idx) {
        const double radius = compute_move_line_blend_radius(poses, idx, requested_blend_radius);
        if (!validate_blend_segment_samples(
                algo,
                poses,
                joint_solutions,
                idx,
                radius,
                error_kind,
                reason,
                ik_ret,
                sample_index)) {
            fail_index = idx;
            return false;
        }
    }
    return true;
}

std::string trim(const std::string &s)
{
    std::string r = s;
    while (!r.empty() && (r.back() == '\r' || r.back() == '\n' || r.back() == ' '))
        r.pop_back();
    while (!r.empty() && (r.front() == ' '))
        r.erase(r.begin());
    return r;
}

const char *servo_ret_name(int code)
{
    switch (code) {
    case 0:
        return "OK";
    case 1:
        return "AUBO_BAD_STATE_WARN";
    case 2:
        return "AUBO_QUEUE_FULL";
    case 3:
        return "AUBO_BUSY";
    case -1:
        return "AUBO_BAD_STATE";
    case -5:
        return "AUBO_INVL_ARGUMENT";
    case -13:
        return "AUBO_REQUEST_IGNORE";
    case -23:
        return "AUBO_IK_NO_CONVERGE";
    case -24:
        return "AUBO_IK_OUT_OF_RANGE";
    case -25:
        return "AUBO_IK_CONFIG_DISMATCH";
    case -26:
        return "AUBO_IK_JACOBIAN_FAILED";
    case -27:
        return "AUBO_IK_NO_SOLU";
    case -28:
        return "AUBO_IK_UNKNOWN_ERROR";
    default:
        return "UNKNOWN";
    }
}

bool solve_linear_system(std::vector<std::vector<double>> a,
                         std::vector<double> b,
                         std::vector<double> &x)
{
    const int n = static_cast<int>(a.size());
    if (n <= 0 || static_cast<int>(b.size()) != n) {
        return false;
    }

    for (int i = 0; i < n; ++i) {
        int pivot = i;
        double pivot_abs = std::abs(a[i][i]);
        for (int r = i + 1; r < n; ++r) {
            const double cand = std::abs(a[r][i]);
            if (cand > pivot_abs) {
                pivot = r;
                pivot_abs = cand;
            }
        }
        if (pivot_abs < 1e-9) {
            return false;
        }
        if (pivot != i) {
            std::swap(a[i], a[pivot]);
            std::swap(b[i], b[pivot]);
        }

        const double diag = a[i][i];
        for (int c = i; c < n; ++c) {
            a[i][c] /= diag;
        }
        b[i] /= diag;

        for (int r = 0; r < n; ++r) {
            if (r == i) {
                continue;
            }
            const double factor = a[r][i];
            if (std::abs(factor) < 1e-12) {
                continue;
            }
            for (int c = i; c < n; ++c) {
                a[r][c] -= factor * a[i][c];
            }
            b[r] -= factor * b[i];
        }
    }

    x = std::move(b);
    return true;
}

bool parse_pose6_and_joint6(const std::string &text,
                            std::vector<double> &pose,
                            double &joint6)
{
    std::istringstream iss(text);
    pose.clear();
    for (int i = 0; i < 6; ++i) {
        double value = 0.0;
        if (!(iss >> value)) {
            return false;
        }
        pose.push_back(value);
    }
    return static_cast<bool>(iss >> joint6);
}

bool solve_pose_roll_pitch_j6_target(RobotAlgorithmPtr algo,
                                     const std::vector<double> &seed_q,
                                     const std::vector<double> &target_pose,
                                     double target_joint6,
                                     std::vector<double> &target_q,
                                     std::vector<double> &fk_pose,
                                     int &jac_ret,
                                     std::string &error)
{
    if (seed_q.size() != 6 || target_pose.size() != 6) {
        error = "invalid_input";
        return false;
    }

    auto fk_result = algo->forwardKinematics(seed_q);
    const auto current_pose = std::get<0>(fk_result);
    const int fk_ret = std::get<1>(fk_result);
    if (fk_ret != 0 || current_pose.size() != 6) {
        jac_ret = fk_ret;
        error = "fk_fail";
        return false;
    }

    auto jac_result = algo->calcJacobian(seed_q, true);
    const auto jac_flat = std::get<0>(jac_result);
    jac_ret = std::get<1>(jac_result);
    if (jac_ret != 0 || jac_flat.size() != 36) {
        error = "jacobian_fail";
        return false;
    }

    std::array<std::array<double, 6>, 6> jacobian{};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            jacobian[r][c] = jac_flat[r * 6 + c];
        }
    }

    std::array<double, 5> rhs{};
    rhs[0] = target_pose[0] - current_pose[0];
    rhs[1] = target_pose[1] - current_pose[1];
    rhs[2] = target_pose[2] - current_pose[2];
    rhs[3] = wrap_angle(target_pose[3] - current_pose[3]);
    rhs[4] = wrap_angle(target_pose[4] - current_pose[4]);

    double dq6 = wrap_angle(target_joint6 - seed_q[5]);
    constexpr double max_joint_step = 0.08;
    dq6 = std::max(-max_joint_step, std::min(max_joint_step, dq6));

    for (int row = 0; row < 5; ++row) {
        rhs[row] -= jacobian[row][5] * dq6;
    }

    std::vector<std::vector<double>> normal(5, std::vector<double>(5, 0.0));
    std::vector<double> rhs_normal(5, 0.0);
    constexpr double lambda = 1e-4;
    for (int r = 0; r < 5; ++r) {
        for (int c = 0; c < 5; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 5; ++k) {
                sum += jacobian[k][r] * jacobian[k][c];
            }
            if (r == c) {
                sum += lambda * lambda;
            }
            normal[r][c] = sum;
        }
        double sum = 0.0;
        for (int k = 0; k < 5; ++k) {
            sum += jacobian[k][r] * rhs[k];
        }
        rhs_normal[r] = sum;
    }

    std::vector<double> dq_head;
    if (!solve_linear_system(normal, rhs_normal, dq_head) || dq_head.size() != 5) {
        error = "linear_solve_fail";
        return false;
    }

    double max_abs_delta = std::abs(dq6);
    for (double value : dq_head) {
        max_abs_delta = std::max(max_abs_delta, std::abs(value));
    }
    if (max_abs_delta > max_joint_step) {
        const double scale = max_joint_step / max_abs_delta;
        for (double &value : dq_head) {
            value *= scale;
        }
        dq6 *= scale;
    }

    target_q = seed_q;
    for (int idx = 0; idx < 5; ++idx) {
        target_q[idx] += dq_head[idx];
    }
    target_q[5] += dq6;

    auto target_fk = algo->forwardKinematics(target_q);
    fk_pose = std::get<0>(target_fk);
    const int target_fk_ret = std::get<1>(target_fk);
    if (target_fk_ret != 0 || fk_pose.size() != 6) {
        jac_ret = target_fk_ret;
        error = "target_fk_fail";
        return false;
    }

    return true;
}

int run_daemon(RobotInterfacePtr robot, RobotConfigPtr config, const Options &opt)
{
    auto state = robot->getRobotState();
    auto motion = robot->getMotionControl();
    auto manage = robot->getRobotManage();
    auto algo = robot->getRobotAlgorithm();

    // Print initial snapshot so caller knows we're ready
    print_snapshot(robot, config);
    std::cout << "DAEMON_READY" << std::endl;

    bool servo_active = false;
    std::vector<double> seed_q;
    std::vector<double> hold_pose;  // last known good pose for keepalive
    std::vector<double> hold_q;     // last known good joint target for keepalive
    bool hold_joint_mode = false;
    int sent = 0;
    double servo_track_time_s = opt.track_time_s;
    double servo_lookahead = 0.0;
    double servo_gain = 150.0;

    auto cleanup_servo = [&]() {
        if (servo_active) {
            motion->setServoModeSelect(0);
            servo_active = false;
            std::cout << "servo_active=0" << std::endl;
        }
    };

    auto servo_keepalive_once = [&]() -> int {
        if (hold_joint_mode && hold_q.size() == 6) {
            return motion->servoJoint(hold_q, 0.0, 0.0, servo_track_time_s, servo_lookahead, servo_gain);
        }
        if (!hold_pose.empty()) {
            return motion->servoCartesian(hold_pose, 0.0, 0.0, servo_track_time_s, servo_lookahead, servo_gain);
        }
        return 0;
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        if (line == "snapshot") {
            print_snapshot(robot, config);
            std::cout << "END" << std::endl;

        } else if (line.rfind("servo_start", 0) == 0) {
            // servo_start [track_time_s]
            // Enter servo mode, keep it open until servo_stop
            cleanup_servo();

            // Optional track_time_s after the command
            {
                std::istringstream iss(line.substr(11));
                double t;
                if (iss >> t && t > 0.0) {
                    servo_track_time_s = t;
                }
            }

            // Ensure robot is ready
            state = robot->getRobotState();
            const int ready_ret = ensure_robot_ready_for_motion(robot, motion, manage);
            if (ready_ret != 0) {
                std::cout << "servo_start_ret=-1" << std::endl;
                std::cout << "error=robot_not_ready" << std::endl;
                std::cout << "ready_ret=" << ready_ret << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }
            state = robot->getRobotState();

            configure_servo_thread_scheduling();

            motion->setSpeedFraction(opt.speed_fraction);
            const double cycle_time_s = config->getCycletime();
            servo_lookahead = std::max(0.03, std::min(0.2, cycle_time_s));
            servo_gain = 150.0;
            int enter_ret = motion->setServoModeSelect(1);
            if (enter_ret != 0 || motion->getServoModeSelect() != 1) {
                std::cout << "servo_start_ret=" << enter_ret << std::endl;
                std::cout << "error=enter_servo_failed" << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }
            servo_active = true;
            seed_q = state->getJointPositions();
            hold_q = seed_q;
            hold_pose = std::get<0>(algo->forwardKinematics(seed_q));
            hold_joint_mode = true;
            sent = 0;
            std::cout << "servo_start_ret=0" << std::endl;
            std::cout << "servo_track_time_s=" << servo_track_time_s << std::endl;
            std::cout << "END" << std::endl;

            // Keepalive immediately after servo_start so the mode stays
            // alive while Python prepares the first chunk.
            if (servo_active && !hold_pose.empty()) {
                while (true) {
                    struct pollfd pfd;
                    pfd.fd = STDIN_FILENO;
                    pfd.events = POLLIN;
                    int wait_ms = std::max(1, (int)(servo_track_time_s * 1000));
                    int pr = poll(&pfd, 1, wait_ms);
                    if (pr > 0) break;
                    servo_keepalive_once();
                }
            }

        } else if (line.rfind("servo_pose ", 0) == 0) {
            // servo_pose x y z rx ry rz
            if (!servo_active) {
                std::cout << "error=servo_not_active" << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }
            std::vector<double> pose;
            if (!parse_pose6(line.substr(11), pose)) {
                std::cout << "error=bad_pose" << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }


            // IK check
            auto ik_result = algo->inverseKinematics(seed_q, pose);
            auto target_q = std::get<0>(ik_result);
            int ik_ret = std::get<1>(ik_result);
            if (ik_ret != 0 || target_q.size() != 6) {
                std::cout << "servo_pose_ret=-1" << std::endl;
                std::cout << "error=ik_fail" << std::endl;
                std::cout << "ik_ret=" << ik_ret << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }
            // Send servo command
            int servo_ret = -1;
            for (int retry = 0; retry < 5; ++retry) {
                servo_ret = motion->servoCartesian(pose, 0.0, 0.0, servo_track_time_s, servo_lookahead, servo_gain);
                if (servo_ret == 0) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            if (servo_ret != 0) {
                std::cout << "servo_pose_ret=" << servo_ret << std::endl;
                std::cout << "servo_ret_name=" << servo_ret_name(servo_ret) << std::endl;
                std::cout << "error=servo_fail" << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }

            seed_q = target_q;
            hold_q = target_q;
            hold_pose = pose;
            hold_joint_mode = false;
            ++sent;

            // Sleep for the servo period
            std::this_thread::sleep_for(std::chrono::duration<double>(servo_track_time_s));

            // Safety check
            state = robot->getRobotState();
            if (state->isCollisionOccurred() || !state->isWithinSafetyLimits()) {
                motion->stopJoint(1.0);
                cleanup_servo();
                std::cout << "servo_pose_ret=-3" << std::endl;
                std::cout << "error=safety" << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }

            std::cout << "servo_pose_ret=0" << std::endl;
            std::cout << "sent=" << sent << std::endl;
            std::cout << "END" << std::endl;

            // Keepalive after servo_pose: hold last pose until next
            // command arrives, preventing servo timeout between calls.
            if (servo_active && !hold_pose.empty()) {
                while (true) {
                    struct pollfd pfd;
                    pfd.fd = STDIN_FILENO;
                    pfd.events = POLLIN;
                    int wait_ms = std::max(1, (int)(servo_track_time_s * 1000));
                    int pr = poll(&pfd, 1, wait_ms);
                    if (pr > 0) break;
                    servo_keepalive_once();
                }
            }

        } else if (line.rfind("servo_pose_j6 ", 0) == 0) {
            // servo_pose_j6 x y z rx ry rz j6
            if (!servo_active) {
                std::cout << "error=servo_not_active" << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }

            std::vector<double> pose;
            double target_joint6 = 0.0;
            if (!parse_pose6_and_joint6(line.substr(14), pose, target_joint6)) {
                std::cout << "error=bad_pose_or_joint6" << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }


            std::vector<double> target_q;
            std::vector<double> fk_pose;
            std::string solve_error;
            int jac_ret = 0;
            if (!solve_pose_roll_pitch_j6_target(
                    algo, seed_q, pose, target_joint6, target_q, fk_pose, jac_ret, solve_error)) {
                std::cout << "servo_pose_ret=-1" << std::endl;
                std::cout << "error=" << solve_error << std::endl;
                std::cout << "ik_ret=" << jac_ret << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }
            int servo_ret = -1;
            for (int retry = 0; retry < 5; ++retry) {
                servo_ret = motion->servoJoint(target_q, 0.0, 0.0, servo_track_time_s, servo_lookahead, servo_gain);
                if (servo_ret == 0) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            if (servo_ret != 0) {
                std::cout << "servo_pose_ret=" << servo_ret << std::endl;
                std::cout << "servo_ret_name=" << servo_ret_name(servo_ret) << std::endl;
                std::cout << "error=servo_fail" << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }

            seed_q = target_q;
            hold_q = target_q;
            hold_pose = fk_pose;
            hold_joint_mode = true;
            ++sent;

            std::this_thread::sleep_for(std::chrono::duration<double>(servo_track_time_s));

            state = robot->getRobotState();
            if (state->isCollisionOccurred() || !state->isWithinSafetyLimits()) {
                motion->stopJoint(1.0);
                cleanup_servo();
                std::cout << "servo_pose_ret=-3" << std::endl;
                std::cout << "error=safety" << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }

            std::cout << "servo_pose_ret=0" << std::endl;
            std::cout << "sent=" << sent << std::endl;
            print_vec("target_q_rad", target_q);
            print_vec("target_pose", fk_pose);
            std::cout << "END" << std::endl;

            if (servo_active && hold_q.size() == 6) {
                while (true) {
                    struct pollfd pfd;
                    pfd.fd = STDIN_FILENO;
                    pfd.events = POLLIN;
                    int wait_ms = std::max(1, (int)(servo_track_time_s * 1000));
                    int pr = poll(&pfd, 1, wait_ms);
                    if (pr > 0) break;
                    servo_keepalive_once();
                }
            }

        } else if (line.rfind("servo_chunk ", 0) == 0) {
            // servo_chunk N
            // Followed by N lines, each containing 6 floats (a pose).
            // Executes all poses in sequence with proper timing, but checks
            // stdin for an "abort" line between poses to allow early exit.
            if (!servo_active) {
                std::cout << "error=servo_not_active" << std::endl;
                std::cout << "END" << std::endl;
                // consume the N pose lines
                int n = 0;
                { std::istringstream iss(line.substr(12)); iss >> n; }
                for (int i = 0; i < n; ++i) {
                    std::string dummy;
                    std::getline(std::cin, dummy);
                }
                continue;
            }
            int num_poses = 0;
            { std::istringstream iss(line.substr(12)); iss >> num_poses; }

            // Read all poses directly — data is either already in the
            // iostream buffer or arrives within microseconds.  Do NOT
            // use poll() here: std::cin may have buffered the lines
            // already, leaving the fd empty, which makes poll() hang.
            std::vector<std::vector<double>> chunk_poses;
            chunk_poses.reserve(num_poses);
            for (int i = 0; i < num_poses; ++i) {
                std::string pline;
                if (!std::getline(std::cin, pline)) break;
                std::vector<double> p;
                if (parse_pose6(trim(pline), p)) {
                    chunk_poses.push_back(p);
                }
            }

            // Execute with proper timing
            int chunk_sent = 0;
            bool chunk_ok = true;
            std::string chunk_error;
            for (size_t idx = 0; idx < chunk_poses.size(); ++idx) {
                auto pose = chunk_poses[idx];
                // IK
                auto ik_result = algo->inverseKinematics(seed_q, pose);
                auto target_q = std::get<0>(ik_result);
                int ik_ret = std::get<1>(ik_result);
                if (ik_ret != 0 || target_q.size() != 6) {
                    chunk_ok = false;
                    chunk_error = "ik_fail";
                    break;
                }
                // Servo
                int servo_ret = -1;
                for (int retry = 0; retry < 5; ++retry) {
                    servo_ret = motion->servoCartesian(pose, 0.0, 0.0, servo_track_time_s, servo_lookahead, servo_gain);
                    if (servo_ret == 0) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                if (servo_ret != 0) {
                    chunk_ok = false;
                    chunk_error = std::string("servo_fail:") + servo_ret_name(servo_ret);
                    std::cout << "servo_ret=" << servo_ret << std::endl;
                    std::cout << "servo_ret_name=" << servo_ret_name(servo_ret) << std::endl;
                    break;
                }
                seed_q = target_q;
                hold_q = target_q;
                hold_pose = pose;
                hold_joint_mode = false;
                ++chunk_sent;
                ++sent;
                std::this_thread::sleep_for(std::chrono::duration<double>(servo_track_time_s));
                // Safety check
                state = robot->getRobotState();
                if (state->isCollisionOccurred() || !state->isWithinSafetyLimits()) {
                    motion->stopJoint(1.0);
                    cleanup_servo();
                    chunk_ok = false;
                    chunk_error = "safety";
                    break;
                }
            }
            std::cout << "servo_chunk_ret=" << (chunk_ok ? 0 : -1) << std::endl;
            std::cout << "chunk_sent=" << chunk_sent << std::endl;
            std::cout << "chunk_total=" << chunk_poses.size() << std::endl;
            if (!chunk_ok) {
                std::cout << "error=" << chunk_error << std::endl;
            }
            std::cout << "END" << std::endl;

            // Keepalive: hold last known good pose until the next command
            // arrives on stdin.  Runs even after non-safety failures (e.g.
            // ik_fail, servo_fail) because the servo mode itself is still
            // active and we don't want it to timeout.
            if (servo_active && chunk_error != "safety") {
                while (true) {
                    struct pollfd pfd;
                    pfd.fd = STDIN_FILENO;
                    pfd.events = POLLIN;
                    int wait_ms = std::max(1, (int)(servo_track_time_s * 1000));
                    int pr = poll(&pfd, 1, wait_ms);
                    if (pr > 0) {
                        break;
                    }
                    servo_keepalive_once();
                }
            }

        } else if (line == "servo_stop") {
            int total_sent = sent;
            cleanup_servo();
            auto final_q = robot->getRobotState()->getJointPositions();
            print_vec("final_q_rad", final_q);
            std::cout << "servo_sent=" << total_sent << std::endl;
            std::cout << "collision_after=" << robot->getRobotState()->isCollisionOccurred() << std::endl;
            std::cout << "within_safety_limits_after=" << robot->getRobotState()->isWithinSafetyLimits() << std::endl;
            std::cout << "END" << std::endl;

        } else if (line.rfind("movel_async_speed ", 0) == 0 ||
                   line.rfind("movel_speed ", 0) == 0 ||
                   line.rfind("movel_async ", 0) == 0 ||
                   line.rfind("movel ", 0) == 0) {
            // movel x y z rx ry rz [speed_frac]
            // movel_async x y z rx ry rz [speed_frac]
            // movel_speed x y z rx ry rz speed_mps
            // movel_async_speed x y z rx ry rz speed_mps
            // Cartesian linear move via moveLine.
            if (servo_active) {
                cleanup_servo();
            }
            const bool use_explicit_speed = line.rfind("movel_async_speed ", 0) == 0 ||
                                            line.rfind("movel_speed ", 0) == 0;
            const bool async_move = line.rfind("movel_async_speed ", 0) == 0 ||
                                    line.rfind("movel_async ", 0) == 0;
            const size_t prefix_len = use_explicit_speed
                ? (async_move ? std::string("movel_async_speed ").size() : std::string("movel_speed ").size())
                : (async_move ? std::string("movel_async ").size() : std::string("movel ").size());
            // Parse: 6 pose values + optional speed
            std::vector<double> pose(6);
            double spd = opt.speed_fraction;
            double line_speed = kDefaultMoveLineSpeed;
            {
                std::istringstream iss(line.substr(prefix_len));
                for (int k = 0; k < 6; ++k) {
                    if (!(iss >> pose[k])) { pose.clear(); break; }
                }
                double s;
                if (iss >> s && s > 0.0) {
                    if (use_explicit_speed) {
                        line_speed = s;
                        spd = line_speed / kDefaultMoveLineSpeed;
                    } else if (s <= 1.0) {
                        spd = s;
                        line_speed = kDefaultMoveLineSpeed * spd;
                    }
                } else if (!use_explicit_speed) {
                    line_speed = kDefaultMoveLineSpeed * spd;
                }
            }
            if (pose.size() != 6) {
                std::cout << "error=bad_pose" << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }
            // Use fresh joint positions as IK seed
            seed_q = robot->getRobotState()->getJointPositions();
            // IK to get target joints
            auto ik_result = algo->inverseKinematics(seed_q, pose);
            auto target_q = std::get<0>(ik_result);
            int ik_ret = std::get<1>(ik_result);
            if (ik_ret != 0 || target_q.size() != 6) {
                std::cout << "movel_ret=-1" << std::endl;
                std::cout << "error=ik_fail" << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }
            // Cancel queued motion first, then wait for steady + ensure Running
            state = robot->getRobotState();
            if (motion->getExecId() != -1 || !state->isSteady()) {
                motion->stopMove(true, true);
                // Wait for robot to become steady after stop (max 500ms)
                auto steady_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
                while (std::chrono::steady_clock::now() < steady_deadline) {
                    state = robot->getRobotState();
                    if (motion->getExecId() == -1 && state->isSteady()) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            }
            // Ensure the controller is cleared and ready before accepting moveLine.
            const int ready_ret = ensure_robot_ready_for_motion(robot, motion, manage);
            if (ready_ret != 0) {
                std::cout << "movel_ret=-1" << std::endl;
                std::cout << "error=robot_not_ready" << std::endl;
                std::cout << "ready_ret=" << ready_ret << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }
            motion->setSpeedFraction(1.0);
            const double line_acc = kDefaultMoveLineAcc;
            int move_ret = 0;
            for (int retry = 0; retry < 40; ++retry) {
                move_ret = motion->moveLine(pose, line_acc, line_speed, 0.0, 0.0);
                if (move_ret == 0) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            std::cout << "movel_ret=" << move_ret << std::endl;
            std::cout << "movel_mode=" << (async_move ? "moveLine_async" : "moveLine") << std::endl;
            std::cout << "requested_speed_fraction=" << spd << std::endl;
            std::cout << "applied_speed_fraction=" << spd << std::endl;
            std::cout << "requested_speed_mps=" << line_speed << std::endl;
            std::cout << "movel_acc=" << line_acc << std::endl;
            std::cout << "movel_speed=" << line_speed << std::endl;
            if (move_ret == 0 && async_move) {
                int exec_id = wait_for_exec_start(robot, 300);
                std::cout << "exec_id=" << exec_id << std::endl;
            } else if (move_ret == 0) {
                int wait_ret = wait_arrival_with_safety(robot, 1.0);
                std::cout << "wait_ret=" << wait_ret << std::endl;
            }
            seed_q = robot->getRobotState()->getJointPositions();
            hold_pose = std::get<0>(algo->forwardKinematics(seed_q));
            print_vec("final_pose", hold_pose);
            std::cout << "END" << std::endl;

        } else if (line.rfind("movel_chunk ", 0) == 0) {
            // movel_chunk N [speed_frac] [blend_radius]
            if (servo_active) {
                cleanup_servo();
            }

            int num_poses = 0;
            double spd = opt.speed_fraction;
            double blend_radius = kDefaultMoveLineBlendRadius;
            {
                std::istringstream iss(line.substr(12));
                iss >> num_poses;
                double parsed_spd = 0.0;
                double parsed_blend = 0.0;
                if (iss >> parsed_spd && parsed_spd > 0.0 && parsed_spd <= 1.0) {
                    spd = parsed_spd;
                }
                if (iss >> parsed_blend && parsed_blend >= 0.0) {
                    blend_radius = parsed_blend;
                }
            }

            std::vector<std::vector<double>> poses;
            poses.reserve(std::max(0, num_poses));
            for (int i = 0; i < num_poses; ++i) {
                std::string pline;
                if (!std::getline(std::cin, pline)) {
                    break;
                }
                std::vector<double> pose;
                if (parse_pose6(trim(pline), pose)) {
                    poses.push_back(pose);
                }
            }

            if (poses.empty()) {
                std::cout << "movel_chunk_ret=-1" << std::endl;
                std::cout << "chunk_total=0" << std::endl;
                std::cout << "chunk_queued=0" << std::endl;
                std::cout << "error=empty_chunk" << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }

            state = robot->getRobotState();
            const int ready_ret = ensure_robot_ready_for_motion(robot, motion, manage);
            if (ready_ret != 0) {
                std::cout << "movel_chunk_ret=-1" << std::endl;
                std::cout << "chunk_total=" << poses.size() << std::endl;
                std::cout << "chunk_queued=0" << std::endl;
                std::cout << "error=robot_not_ready" << std::endl;
                std::cout << "ready_ret=" << ready_ret << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }

            std::vector<double> check_seed_q = robot->getRobotState()->getJointPositions();
            std::vector<std::vector<double>> joint_solutions;
            joint_solutions.reserve(poses.size());
            for (size_t idx = 0; idx < poses.size(); ++idx) {
                auto ik_result = algo->inverseKinematics(check_seed_q, poses[idx]);
                auto target_q = std::get<0>(ik_result);
                int ik_ret = std::get<1>(ik_result);
                if (ik_ret != 0 || target_q.size() != 6) {
                    std::cout << "movel_chunk_ret=-1" << std::endl;
                    std::cout << "chunk_total=" << poses.size() << std::endl;
                    std::cout << "chunk_queued=0" << std::endl;
                    std::cout << "error=ik_fail" << std::endl;
                    std::cout << "ik_ret=" << ik_ret << std::endl;
                    std::cout << "fail_index=" << idx << std::endl;
                    std::cout << "END" << std::endl;
                    poses.clear();
                    break;
                }
                check_seed_q = target_q;
                joint_solutions.push_back(target_q);
            }
            if (poses.empty()) {
                continue;
            }

            std::string swept_error_kind;
            std::string swept_reason;
            int swept_ik_ret = 0;
            size_t swept_fail_index = 0;
            size_t swept_sample_index = 0;
            if (!validate_movel_chunk_swept_path(
                    algo,
                    poses,
                    joint_solutions,
                    blend_radius,
                    swept_error_kind,
                    swept_reason,
                    swept_ik_ret,
                    swept_fail_index,
                    swept_sample_index)) {
                std::cout << "movel_chunk_ret=-1" << std::endl;
                std::cout << "chunk_total=" << poses.size() << std::endl;
                std::cout << "chunk_queued=0" << std::endl;
                std::cout << "error=swept_path_fail" << std::endl;
                std::cout << "swept_error_kind=" << swept_error_kind << std::endl;
                std::cout << "swept_reason=" << swept_reason << std::endl;
                std::cout << "ik_ret=" << swept_ik_ret << std::endl;
                std::cout << "fail_index=" << swept_fail_index << std::endl;
                std::cout << "sample_index=" << swept_sample_index << std::endl;
                std::cout << "END" << std::endl;
                continue;
            }

            motion->setSpeedFraction(1.0);
            const double line_acc = kDefaultMoveLineAcc;
            const double line_speed = kDefaultMoveLineSpeed;
            int chunk_queued = 0;
            int last_ret = 0;
            int exec_id = -1;

            for (size_t idx = 0; idx < poses.size(); ++idx) {
                const double radius = compute_move_line_blend_radius(poses, idx, blend_radius);
                last_ret = 0;
                for (int retry = 0; retry < 20; ++retry) {
                    last_ret = motion->moveLine(poses[idx], line_acc, line_speed, radius, 0.0);
                    if (last_ret == 0) {
                        break;
                    }
                    if (last_ret != 2 && last_ret != 3) {
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if (last_ret != 0) {
                    break;
                }
                ++chunk_queued;
                if (chunk_queued == 1) {
                    exec_id = wait_for_exec_start(robot, 300);
                }
            }

            seed_q = robot->getRobotState()->getJointPositions();
            hold_pose = std::get<0>(algo->forwardKinematics(seed_q));
            std::cout << "movel_chunk_ret=" << (last_ret == 0 && chunk_queued == static_cast<int>(poses.size()) ? 0 : last_ret) << std::endl;
            std::cout << "chunk_total=" << poses.size() << std::endl;
            std::cout << "chunk_queued=" << chunk_queued << std::endl;
            std::cout << "exec_id=" << exec_id << std::endl;
            std::cout << "movel_mode=moveLine_chunk" << std::endl;
            std::cout << "requested_speed_fraction=" << spd << std::endl;
            std::cout << "applied_speed_fraction=1" << std::endl;
            std::cout << "movel_acc=" << line_acc << std::endl;
            std::cout << "movel_speed=" << line_speed << std::endl;
            std::cout << "movel_blend_radius=" << blend_radius << std::endl;
            if (last_ret != 0 && chunk_queued != static_cast<int>(poses.size())) {
                std::cout << "error=queue_failed" << std::endl;
                std::cout << "motion_ret_name=" << servo_ret_name(last_ret) << std::endl;
                std::cout << "fail_index=" << chunk_queued << std::endl;
            }
            print_vec("final_pose", hold_pose);
            std::cout << "END" << std::endl;

        } else if (line == "motion_status") {
            auto robot_state = robot->getRobotState();
            std::cout << "exec_id=" << motion->getExecId() << std::endl;
            std::cout << "queue_size=" << motion->getTrajectoryQueueSize() << std::endl;
            std::cout << "is_steady=" << robot_state->isSteady() << std::endl;
            std::cout << "robot_mode=" << robot_state->getRobotModeType() << std::endl;
            std::cout << "safety_mode=" << robot_state->getSafetyModeType() << std::endl;
            std::cout << "collision=" << robot_state->isCollisionOccurred() << std::endl;
            std::cout << "within_safety_limits=" << robot_state->isWithinSafetyLimits() << std::endl;
            print_vec("joint_q_rad", robot_state->getJointPositions());
            print_vec("tcp_pose", robot_state->getTcpPose());
            std::cout << "END" << std::endl;

        } else if (line == "wait_motion") {
            auto robot_state = robot->getRobotState();
            int wait_ret = 0;
            if (!(motion->getExecId() == -1 && robot_state->isSteady())) {
                wait_ret = wait_arrival_with_safety(robot, 1.0);
            }
            std::cout << "wait_ret=" << wait_ret << std::endl;
            std::cout << "exec_id=" << motion->getExecId() << std::endl;
            print_vec("final_q_rad", robot_state->getJointPositions());
            print_vec("final_pose", robot_state->getTcpPose());
            std::cout << "collision_after=" << robot_state->isCollisionOccurred() << std::endl;
            std::cout << "within_safety_limits_after=" << robot_state->isWithinSafetyLimits() << std::endl;
            std::cout << "robot_mode_after=" << robot_state->getRobotModeType() << std::endl;
            std::cout << "safety_mode_after=" << robot_state->getSafetyModeType() << std::endl;
            std::cout << "END" << std::endl;

        } else if (line.rfind("stop_motion", 0) == 0) {
            bool quick = true;
            bool all_tasks = true;
            {
                std::istringstream iss(line.substr(11));
                int quick_i = 1;
                int all_tasks_i = 1;
                if (iss >> quick_i) {
                    quick = quick_i != 0;
                }
                if (iss >> all_tasks_i) {
                    all_tasks = all_tasks_i != 0;
                }
            }

            if (servo_active) {
                cleanup_servo();
            }

            auto robot_state = robot->getRobotState();
            int stop_ret = 0;
            if (!(motion->getExecId() == -1 && robot_state->isSteady())) {
                stop_ret = motion->stopMove(quick, all_tasks);
            }

            int wait_ret = 0;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (std::chrono::steady_clock::now() < deadline) {
                robot_state = robot->getRobotState();
                if (motion->getExecId() == -1 && robot_state->isSteady()) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            robot_state = robot->getRobotState();
            if (!(motion->getExecId() == -1 && robot_state->isSteady())) {
                wait_ret = -3;
            }

            std::cout << "stop_ret=" << stop_ret << std::endl;
            std::cout << "wait_ret=" << wait_ret << std::endl;
            std::cout << "exec_id=" << motion->getExecId() << std::endl;
            std::cout << "is_steady=" << robot_state->isSteady() << std::endl;
            print_vec("final_q_rad", robot_state->getJointPositions());
            print_vec("final_pose", robot_state->getTcpPose());
            std::cout << "collision_after=" << robot_state->isCollisionOccurred() << std::endl;
            std::cout << "within_safety_limits_after=" << robot_state->isWithinSafetyLimits() << std::endl;
            std::cout << "robot_mode_after=" << robot_state->getRobotModeType() << std::endl;
            std::cout << "safety_mode_after=" << robot_state->getSafetyModeType() << std::endl;
            std::cout << "END" << std::endl;

        } else if (line == "quit" || line == "exit") {
            cleanup_servo();
            break;
        } else {
            std::cout << "error=unknown_command" << std::endl;
            std::cout << "END" << std::endl;
        }
    }
    cleanup_servo();
    return 0;
}

}  // namespace

int main(int argc, char **argv)
{
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        return 1;
    }

    auto cli = std::make_shared<RpcClient>();
    cli->setRequestTimeout(2500);
    int conn = cli->connect(opt.robot_ip, opt.port);
    std::cout << "connect_ret=" << conn << std::endl;
    std::cout << "hasConnected=" << cli->hasConnected() << std::endl;
    if (conn != 0 || !cli->hasConnected()) {
        return 2;
    }

    int login = cli->login(opt.user, opt.password);
    std::cout << "login_ret=" << login << std::endl;
    std::cout << "hasLogined=" << cli->hasLogined() << std::endl;
    if (login != 0 || !cli->hasLogined()) {
        cli->disconnect();
        return 3;
    }

    auto robot_name = cli->getRobotNames().front();
    auto robot = cli->getRobotInterface(robot_name);
    auto state = robot->getRobotState();
    auto config = robot->getRobotConfig();
    auto motion = robot->getMotionControl();
    auto manage = robot->getRobotManage();
    auto algo = robot->getRobotAlgorithm();

    // ---- Daemon mode: stay connected, serve snapshot + servo from stdin ----
    if (opt.daemon) {
        int rc = run_daemon(robot, config, opt);
        cli->logout();
        cli->disconnect();
        return rc;
    }

    // ---- One-shot mode (original behavior) ----
    auto joint_q = state->getJointPositions();
    auto tcp_pose = state->getTcpPose();
    auto tool_pose = state->getToolPose();
    auto tcp_offset = state->getActualTcpOffset();
    auto elbow_pos = state->getElbowPosistion();

    std::cout << "robot_name=" << robot_name << std::endl;
    std::cout << "robot_mode=" << state->getRobotModeType() << std::endl;
    std::cout << "safety_mode=" << state->getSafetyModeType() << std::endl;
    std::cout << "is_power_on=" << state->isPowerOn() << std::endl;
    std::cout << "collision=" << state->isCollisionOccurred() << std::endl;
    std::cout << "within_safety_limits=" << state->isWithinSafetyLimits() << std::endl;
    std::cout << "collision_level=" << config->getCollisionLevel() << std::endl;
    std::cout << "collision_stop_type=" << config->getCollisionStopType() << std::endl;
    print_vec("joint_q_rad", joint_q);
    print_vec("tcp_pose", tcp_pose);
    print_vec("tool_pose", tool_pose);
    print_vec("tcp_offset", tcp_offset);
    print_vec("elbow_pos", elbow_pos);

    std::vector<double> target_q;
    int ik_ret = 0;
    if (opt.has_target_pose) {
        print_vec("target_pose", opt.target_pose);
        auto ik_result = algo->inverseKinematics(joint_q, opt.target_pose);
        target_q = std::get<0>(ik_result);
        ik_ret = std::get<1>(ik_result);
        std::cout << "ik_ret=" << ik_ret << std::endl;
        if (!target_q.empty()) {
            print_vec("target_q_rad", target_q);
        }
        if (ik_ret != 0) {
            cli->logout();
            cli->disconnect();
            return 20;
        }
    }

    if (opt.has_joint_target) {
        target_q = opt.joint_target;
        print_vec("target_q_rad", target_q);
    }

    if (!opt.execute) {
        if (opt.has_track_pose_file) {
            try {
                const auto poses = read_pose_file(opt.track_pose_file);
                std::cout << "track_pose_file=" << opt.track_pose_file << std::endl;
                std::cout << "track_steps=" << poses.size() << std::endl;
            } catch (const std::exception &e) {
                std::cerr << e.what() << std::endl;
                cli->logout();
                cli->disconnect();
                return 27;
            }
        }
        cli->logout();
        cli->disconnect();
        return 0;
    }

    const int ready_ret = ensure_robot_ready_for_motion(robot, motion, manage);
    if (ready_ret != 0) {
        cli->logout();
        cli->disconnect();
        return 23;
    }

    if (opt.has_track_pose_file) {
        const int track_rc = run_track_pose_sequence(robot, motion, algo, config, opt);
        cli->logout();
        cli->disconnect();
        return track_rc;
    }

    if (target_q.empty()) {
        std::cerr << "execute requested but no target_q available" << std::endl;
        cli->logout();
        cli->disconnect();
        return 21;
    }

    const double speed_rad = opt.speed_deg / 180.0 * kPi;
    const double acc_rad = opt.acc_deg / 180.0 * kPi;
    std::cout << "speed_deg=" << opt.speed_deg << std::endl;
    std::cout << "acc_deg=" << opt.acc_deg << std::endl;
    std::cout << "speed_fraction=" << opt.speed_fraction << std::endl;

    motion->setSpeedFraction(opt.speed_fraction);
    int move_ret = motion->moveJoint(target_q, acc_rad, speed_rad, 0.0, 0.0);
    std::cout << "moveJoint_ret=" << move_ret << std::endl;
    if (move_ret != 0) {
        cli->logout();
        cli->disconnect();
        return 25;
    }

    int wait_ret = wait_arrival_with_safety(robot, 1.0);
    std::cout << "wait_arrival_ret=" << wait_ret << std::endl;

    auto final_q = state->getJointPositions();
    print_vec("final_q_rad", final_q);
    std::cout << "collision_after=" << state->isCollisionOccurred() << std::endl;
    std::cout << "within_safety_limits_after=" << state->isWithinSafetyLimits() << std::endl;
    std::cout << "robot_mode_after=" << state->getRobotModeType() << std::endl;
    std::cout << "safety_mode_after=" << state->getSafetyModeType() << std::endl;

    cli->logout();
    cli->disconnect();
    return (wait_ret == 0) ? 0 : 26;
}
