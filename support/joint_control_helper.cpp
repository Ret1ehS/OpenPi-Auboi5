#include "aubo_sdk/rpc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace arcs::aubo_sdk;
using namespace arcs::common_interface;

namespace {

constexpr double kPi = 3.14159265358979323846;
struct Options {
    std::string robot_ip = "192.168.1.100";
    int port = 30004;
    std::string user = "aubo";
    std::string password = "123456";
    double speed_deg = 10.0;
    double acc_deg = 20.0;
    double speed_fraction = 1.0;
    bool execute = false;
    std::vector<double> joint_target;
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
        } else if (arg == "--joint-target") {
            opt.joint_target.clear();
            for (int j = 0; j < 6; ++j) {
                opt.joint_target.push_back(std::stod(require_value("--joint-target")));
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: aubo_joint_control_helper [--execute] [--robot-ip IP] [--port 30004]\n"
                << "                                 [--user aubo] [--password 123456]\n"
                << "                                 [--speed-deg 10] [--acc-deg 20]\n"
                << "                                 [--speed-fraction 1.0]\n"
                << "                                 --joint-target q1 q2 q3 q4 q5 q6\n";
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

int preclear_motion_queue(RobotInterfacePtr robot, MotionControlPtr motion, const char *phase, bool force_stop)
{
    auto state = robot->getRobotState();
    const int exec_id_before = motion->getExecId();
    const bool steady_before = state->isSteady();
    const bool should_stop = force_stop || exec_id_before != -1 || !steady_before;

    std::cout << phase << "_exec_id_before=" << exec_id_before << std::endl;
    std::cout << phase << "_steady_before=" << steady_before << std::endl;
    std::cout << phase << "_should_stop=" << should_stop << std::endl;

    int stop_ret = 0;
    if (should_stop) {
        stop_ret = motion->stopMove(true, true);
    }
    const int clear_path_ret = motion->clearPath();
    const int wait_idle_ret = wait_for_motion_idle(robot, motion, 1000);

    std::cout << phase << "_stop_move_ret=" << stop_ret << std::endl;
    std::cout << phase << "_clear_path_ret=" << clear_path_ret << std::endl;
    std::cout << phase << "_wait_idle_ret=" << wait_idle_ret << std::endl;

    state = robot->getRobotState();
    std::cout << phase << "_exec_id_after=" << motion->getExecId() << std::endl;
    std::cout << phase << "_steady_after=" << state->isSteady() << std::endl;
    return wait_idle_ret;
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
    int clear_ret = clear_abnormal_state_sdk(robot, motion, manage);
    std::cout << "clear_abnormal_ret=" << clear_ret << std::endl;

    auto state = robot->getRobotState();
    if (!state->isPowerOn()) {
        int pret = manage->poweron();
        std::cout << "poweron_ret=" << pret << std::endl;
        if (wait_for_robot_mode(robot, RobotModeType::Idle, 10000) != 0 && !state->isPowerOn()) {
            return -1;
        }
        state = robot->getRobotState();
    }

    if (state->getRobotModeType() != RobotModeType::Running) {
        int sret = manage->startup();
        std::cout << "startup_ret=" << sret << std::endl;
        if (wait_for_robot_mode(robot, RobotModeType::Running, 15000) != 0) {
            return -2;
        }
        state = robot->getRobotState();
    }

    return robot_state_motion_ready(state) ? 0 : -3;
}

}  // namespace

int main(int argc, char **argv)
{
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        return 1;
    }
    if (opt.joint_target.size() != 6) {
        std::cerr << "joint_target must contain 6 values" << std::endl;
        return 2;
    }

    auto rpc_cli = std::make_shared<RpcClient>();
    rpc_cli->setRequestTimeout(2500);

    int conn = rpc_cli->connect(opt.robot_ip, opt.port);
    std::cout << "connect_ret=" << conn << std::endl;
    std::cout << "hasConnected=" << rpc_cli->hasConnected() << std::endl;
    if (conn != 0 || !rpc_cli->hasConnected()) {
        return 3;
    }

    int login = rpc_cli->login(opt.user, opt.password);
    std::cout << "login_ret=" << login << std::endl;
    std::cout << "hasLogined=" << rpc_cli->hasLogined() << std::endl;
    if (login != 0 || !rpc_cli->hasLogined()) {
        rpc_cli->disconnect();
        return 4;
    }

    auto robot_name = rpc_cli->getRobotNames().front();
    auto robot = rpc_cli->getRobotInterface(robot_name);
    auto state = robot->getRobotState();
    auto manage = robot->getRobotManage();
    auto motion = robot->getMotionControl();
    auto config = robot->getRobotConfig();

    auto current = state->getJointPositions();
    std::vector<double> delta(current.size(), 0.0);
    for (size_t i = 0; i < current.size() && i < opt.joint_target.size(); ++i) {
        delta[i] = opt.joint_target[i] - current[i];
    }

    std::cout << "robot_name=" << robot_name << std::endl;
    std::cout << "isPowerOn=" << state->isPowerOn() << std::endl;
    std::cout << "robot_mode=" << state->getRobotModeType() << std::endl;
    std::cout << "collision_level=" << config->getCollisionLevel() << std::endl;
    std::cout << "collision_stop_type=" << config->getCollisionStopType() << std::endl;
    std::cout << "collision=" << state->isCollisionOccurred() << std::endl;
    std::cout << "within_safety_limits=" << state->isWithinSafetyLimits() << std::endl;
    print_vec("current_q_rad", current);
    print_vec("target_q_rad", opt.joint_target);
    print_vec("delta_q_rad", delta);

    if (!opt.execute) {
        std::cout << "DRY_RUN_ONLY=1" << std::endl;
        rpc_cli->logout();
        rpc_cli->disconnect();
        return 0;
    }

    const int ready_ret = ensure_robot_ready_for_motion(robot, motion, manage);
    if (ready_ret != 0) {
        std::cerr << "ABORT: robot is not ready for moveJoint after SDK clear." << std::endl;
        rpc_cli->logout();
        rpc_cli->disconnect();
        return 6;
    }

    const double speed_rad = opt.speed_deg / 180.0 * kPi;
    const double acc_rad = opt.acc_deg / 180.0 * kPi;
    std::cout << "speed_deg=" << opt.speed_deg << std::endl;
    std::cout << "acc_deg=" << opt.acc_deg << std::endl;
    std::cout << "speed_fraction=" << opt.speed_fraction << std::endl;

    // Clear any leftover queued path from prior async motion before moveJoint.
    preclear_motion_queue(robot, motion, "pre_move", false);

    motion->setSpeedFraction(opt.speed_fraction);
    int move_ret = motion->moveJoint(opt.joint_target, acc_rad, speed_rad, 0.0, 0.0);
    std::cout << "moveJoint_first_ret=" << move_ret << std::endl;
    if (move_ret != 0) {
        std::cout << "moveJoint_recovery_retry=1" << std::endl;
        preclear_motion_queue(robot, motion, "recovery_retry", true);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        move_ret = motion->moveJoint(opt.joint_target, acc_rad, speed_rad, 0.0, 0.0);
        std::cout << "moveJoint_retry_ret=" << move_ret << std::endl;
    }
    std::cout << "moveJoint_ret=" << move_ret << std::endl;
    if (move_ret != 0) {
        rpc_cli->logout();
        rpc_cli->disconnect();
        return 8;
    }

    int wait_ret = wait_arrival_with_safety(robot, 1.0);
    std::cout << "wait_arrival_ret=" << wait_ret << std::endl;

    auto final_q = state->getJointPositions();
    std::vector<double> final_err(final_q.size(), 0.0);
    double max_abs_err = 0.0;
    for (size_t i = 0; i < final_q.size() && i < opt.joint_target.size(); ++i) {
        final_err[i] = final_q[i] - opt.joint_target[i];
        max_abs_err = std::max(max_abs_err, std::abs(final_err[i]));
    }
    print_vec("final_q_rad", final_q);
    print_vec("final_err_rad", final_err);
    std::cout << "max_abs_err_rad=" << max_abs_err << std::endl;
    std::cout << "collision_after=" << state->isCollisionOccurred() << std::endl;
    std::cout << "within_safety_limits_after=" << state->isWithinSafetyLimits() << std::endl;

    rpc_cli->logout();
    rpc_cli->disconnect();
    return (wait_ret == 0) ? 0 : 9;
}
