#include "aubo_sdk/rpc.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace arcs::aubo_sdk;
using namespace arcs::common_interface;

namespace {

struct Options {
    std::string robot_ip = "192.168.1.100";
    int port = 30004;
    std::string user = "aubo";
    std::string password = "123456";
    bool status = false;
    bool has_set_voltage = false;
    int set_voltage = 0;
    bool has_set_output_mode = false;
    int output_mode_index = 0;
    bool output_mode_input = false;
    bool has_set_output = false;
    int output_index = 0;
    bool output_value = false;
};

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
        } else if (arg == "--status") {
            opt.status = true;
        } else if (arg == "--set-voltage") {
            opt.has_set_voltage = true;
            opt.set_voltage = std::stoi(require_value("--set-voltage"));
        } else if (arg == "--set-output-mode") {
            opt.has_set_output_mode = true;
            opt.output_mode_index = std::stoi(require_value("--set-output-mode"));
            opt.output_mode_input = std::stoi(require_value("--set-output-mode")) != 0;
        } else if (arg == "--set-output") {
            opt.has_set_output = true;
            opt.output_index = std::stoi(require_value("--set-output"));
            opt.output_value = std::stoi(require_value("--set-output")) != 0;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: tool_io_helper [--robot-ip IP] [--port 30004] [--user aubo] [--password 123456]\n"
                << "                      [--status]\n"
                << "                      [--set-voltage 0|12|24]\n"
                << "                      [--set-output-mode INDEX INPUT(0|1)]\n"
                << "                      [--set-output INDEX VALUE(0|1)]\n";
            return false;
        } else {
            std::cerr << "unknown argument: " << arg << std::endl;
            return false;
        }
    }
    return true;
}

void print_status(IoControlPtr io)
{
    const int di_num = io->getToolDigitalInputNum();
    const int do_num = io->getToolDigitalOutputNum();
    const int voltage = io->getToolVoltageOutputDomain();

    std::cout << "tool_di_num=" << di_num << std::endl;
    std::cout << "tool_do_num=" << do_num << std::endl;
    std::cout << "tool_voltage_domain=" << voltage << std::endl;

    const int io_count = di_num > do_num ? di_num : do_num;
    for (int idx = 0; idx < io_count; ++idx) {
        std::cout << "tool_io_" << idx << "_is_input=" << io->isToolIoInput(idx) << std::endl;
        if (idx < di_num) {
            std::cout << "tool_di_" << idx << "=" << io->getToolDigitalInput(idx) << std::endl;
        }
        if (idx < do_num) {
            std::cout << "tool_do_" << idx << "_runstate=" << static_cast<int>(io->getToolDigitalOutputRunstate(idx))
                      << std::endl;
            std::cout << "tool_do_" << idx << "=" << io->getToolDigitalOutput(idx) << std::endl;
        }
    }
}

}  // namespace

int main(int argc, char **argv)
{
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        return 0;
    }

    try {
        auto rpc_cli = std::make_shared<RpcClient>();

        const int connect_ret = rpc_cli->connect(opt.robot_ip, opt.port);
        std::cout << "connect_ret=" << connect_ret << std::endl;
        std::cout << "hasConnected=" << rpc_cli->hasConnected() << std::endl;
        if (connect_ret != 0 || !rpc_cli->hasConnected()) {
            return 2;
        }

        const int login_ret = rpc_cli->login(opt.user, opt.password);
        std::cout << "login_ret=" << login_ret << std::endl;
        std::cout << "hasLogined=" << rpc_cli->hasLogined() << std::endl;
        if (login_ret != 0 || !rpc_cli->hasLogined()) {
            return 3;
        }

        auto api = std::static_pointer_cast<AuboApi>(rpc_cli);
        const auto robot_names = api->getRobotNames();
        if (robot_names.empty()) {
            std::cerr << "no robot names returned by controller" << std::endl;
            return 4;
        }
        const auto robot_name = robot_names.front();
        std::cout << "robot_name=" << robot_name << std::endl;

        auto robot = api->getRobotInterface(robot_name);
        auto io = robot->getIoControl();

        if (opt.has_set_voltage) {
            const int ret = io->setToolVoltageOutputDomain(opt.set_voltage);
            std::cout << "set_voltage_ret=" << ret << std::endl;
        }

        if (opt.has_set_output_mode) {
            const int runstate_ret = io->setToolDigitalOutputRunstate(
                opt.output_mode_index, StandardOutputRunState::None);
            std::cout << "set_output_runstate_ret=" << runstate_ret << std::endl;
            const int ret = io->setToolIoInput(opt.output_mode_index, opt.output_mode_input);
            std::cout << "set_output_mode_ret=" << ret << std::endl;
            std::cout << "tool_io_" << opt.output_mode_index << "_is_input=" << io->isToolIoInput(opt.output_mode_index)
                      << std::endl;
            std::cout << "tool_do_" << opt.output_mode_index << "_runstate="
                      << static_cast<int>(io->getToolDigitalOutputRunstate(opt.output_mode_index)) << std::endl;
        }

        if (opt.has_set_output) {
            const int ret = io->setToolDigitalOutput(opt.output_index, opt.output_value);
            std::cout << "set_output_ret=" << ret << std::endl;
            std::cout << "tool_do_" << opt.output_index << "=" << io->getToolDigitalOutput(opt.output_index)
                      << std::endl;
        }

        if (opt.status || !opt.has_set_output || !opt.has_set_output_mode || !opt.has_set_voltage) {
            print_status(io);
        }

        rpc_cli->logout();
        rpc_cli->disconnect();
        return 0;
    } catch (const std::exception &exc) {
        std::cerr << "exception=" << exc.what() << std::endl;
        return 10;
    }
}
