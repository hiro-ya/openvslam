#ifdef USE_PANGOLIN_VIEWER
#include <pangolin_viewer/viewer.h>
#elif USE_SOCKET_PUBLISHER
#include <socket_publisher/publisher.h>
#endif

#include <openvslam/system.h>
#include <openvslam/config.h>

#include <iostream>
#include <chrono>
#include <numeric>
#include <math.h>

#include <Eigen/Core>

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf_conversions/tf_eigen.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>

#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif

static ros::Publisher openvslam_pose_publisher;
static geometry_msgs::PoseStamped openvslam_pose_msg;

void mono_localization(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path,
                   const std::string& mask_img_path, const std::string& map_db_path, const bool mapping) {
    // load the mask image
    const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);
    // load the prebuilt map
    SLAM.load_map_database(map_db_path);
    // startup the SLAM process (it does not need initialization of a map)
    SLAM.startup(false);
    // select to activate the mapping module or not
    if (mapping) {
        SLAM.enable_mapping_module();
    }
    else {
        SLAM.disable_mapping_module();
    }

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
    std::vector<double> track_times;
    const auto tp_0 = std::chrono::steady_clock::now();

    // initialize this node
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    // initialize publisher
    openvslam_pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("/openvslam_pose", 10);

    // run the SLAM as subscriber
    image_transport::Subscriber sub = it.subscribe("camera/image_raw", 1, [&](const sensor_msgs::ImageConstPtr& msg) {
        const ros::Time rostime = ros::Time::now();
        const auto tp_1 = std::chrono::steady_clock::now();
        const auto timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(tp_1 - tp_0).count();

        // input the current frame and estimate the camera pose
        Eigen::Matrix4d cam_pose = SLAM.track_for_monocular(cv_bridge::toCvShare(msg, "bgr8")->image, timestamp, mask);

        if(SLAM.tracker_is_tracking()) {
            // convert eigen matrix to ros tf
            Eigen::Affine3d cam_pose_affine;
            cam_pose_affine = cam_pose;
            tf::Transform cam_pose_tf;
            tf::transformEigenToTF(cam_pose_affine, cam_pose_tf);

            // transform and broadcast tf
            tf::Transform tf_ovslam_to_world;
            tf_ovslam_to_world.setIdentity();
            tf_ovslam_to_world.setRotation(cam_pose_tf.getRotation()*tf::createQuaternionFromRPY(0.5*M_PI, 0, 0));
            tf_ovslam_to_world.setOrigin(cam_pose_tf.getOrigin());

            // openvslam_pose publisher
            openvslam_pose_msg.header.frame_id = "/world";
            openvslam_pose_msg.header.stamp = rostime;
            openvslam_pose_msg.pose.position.x = tf_ovslam_to_world.inverse().getOrigin().getX();
            openvslam_pose_msg.pose.position.y = tf_ovslam_to_world.inverse().getOrigin().getY();
            openvslam_pose_msg.pose.position.z = tf_ovslam_to_world.inverse().getOrigin().getZ();
            openvslam_pose_msg.pose.orientation.x = tf_ovslam_to_world.inverse().getRotation().x();
            openvslam_pose_msg.pose.orientation.y = tf_ovslam_to_world.inverse().getRotation().y();
            openvslam_pose_msg.pose.orientation.z = tf_ovslam_to_world.inverse().getRotation().z();
            openvslam_pose_msg.pose.orientation.w = tf_ovslam_to_world.inverse().getRotation().w();

            openvslam_pose_publisher.publish(openvslam_pose_msg);
        }

        const auto tp_2 = std::chrono::steady_clock::now();

        const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
        track_times.push_back(track_time);
    });

    ros::spin();

    // shutdown the SLAM process
    SLAM.shutdown();

    if (track_times.size()) {
        std::sort(track_times.begin(), track_times.end());
        const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
        std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
        std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
    }
}

int main(int argc, char* argv[]) {
#ifdef USE_STACK_TRACE_LOGGER
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
#endif
    ros::init(argc, argv, "run_localization");

    // create options
    popl::OptionParser op("Allowed options");
    auto help = op.add<popl::Switch>("h", "help", "produce help message");
    auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
    auto setting_file_path = op.add<popl::Value<std::string>>("c", "config", "setting file path");
    auto map_db_path = op.add<popl::Value<std::string>>("p", "map-db", "path to a prebuilt map database");
    auto mapping = op.add<popl::Switch>("", "mapping", "perform mapping as well as localization");
    auto mask_img_path = op.add<popl::Value<std::string>>("", "mask", "mask image path", "");
    auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
    try {
        op.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // check validness of options
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!vocab_file_path->is_set() || !setting_file_path->is_set() || !map_db_path->is_set()) {
        std::cerr << "invalid arguments" << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
    if (debug_mode->is_set()) {
        spdlog::set_level(spdlog::level::debug);
    }
    else {
        spdlog::set_level(spdlog::level::info);
    }

    // load configuration
    std::shared_ptr<openvslam::config> cfg;
    try {
        cfg = std::make_shared<openvslam::config>(setting_file_path->value());
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStart("slam.prof");
#endif

    // run localization
    if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Monocular) {
        mono_localization(cfg, vocab_file_path->value(), mask_img_path->value(), map_db_path->value(), mapping->is_set());
    }
    else {
        throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif

    return EXIT_SUCCESS;
}
