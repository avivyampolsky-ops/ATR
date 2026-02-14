#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "cpp_modules/hpp/register_detect.hpp"
#include "cpp_modules/hpp/kalman.hpp"

namespace cpp_utils {

struct Config {
    std::unordered_map<std::string, std::string> values;
    std::unordered_map<std::string, std::string> runtime;

    bool load(const std::string &path);
    void set_runtime(const std::string &key, const std::string &value);
    void set_runtime(const std::string &key, int value);
    void set_runtime(const std::string &key, float value);
    std::string get_string(const std::string &key, const std::string &fallback) const;
    bool get_bool(const std::string &key, bool fallback) const;
    int get_int(const std::string &key, int fallback) const;
    float get_float(const std::string &key, float fallback) const;
};

struct DataStreamer {
    std::string data_path;
    int start_frame = 0;
    std::optional<int> end_frame;
    bool input_is_gray = false;
    float fps_override = 30.0f;

    cv::VideoCapture cap;
    std::vector<std::string> image_paths;
    std::optional<int> total_frames;
    bool resolved_input_is_gray = false;
    float fps = 0.0f;
    cv::Mat ref;

    bool init_source();
    cv::Mat read_image(const std::string &path) const;
    cv::Mat get_frame(int frame_idx);
    void close();
};

struct DebugUtilsCpp {
    std::string registration_mode;
    int start_frame = 0;
    std::optional<int> end_frame;
    std::optional<int> total_frames;
    std::string video_base;
    std::string out_dir;
    bool draw_output = true;
    bool debug_mode = true;
    bool enable_timing = true;
    bool write_empty_masks = false;
    int track_min_age = 3;
    int incrimination_thresh = -1;

    std::string frames_dir;
    std::string mog2_dir;
    std::string bb_dir;
    std::string reanchor_log_path;

    float stream_fps = 30.0f;
    bool stream_is_gray = false;

    cv::VideoWriter writer;
    bool writer_warned = false;
    std::string writer_path;
    cv::Size writer_size;
    bool writer_is_color = true;
    int registration_margin = 0;

    int loop_calls = 0;
    std::ofstream homography_log;
    std::ofstream handshake_log;
    int homography_min_inliers = 0;

    DebugUtilsCpp(const Config &config,
                  const std::string &data_path,
                  const std::string &out_root,
                  float stream_fps_in,
                  bool stream_is_gray_in);

    void set_stream_info(const DataStreamer &streamer);
    void close();
    void write_homography_log(int frame_idx, const HomographyRegistrationDebug *homography_debug);
    void set_shape(const cv::Size &shape);
    void note_loop_call();
    void write_reference_events_graph(const std::vector<int> &ref_events);

    cv::Mat annotate_registration_debug(const cv::Mat &frame,
                                        int frame_idx,
                                        const std::set<int> &ref_events,
                                        const HomographyRegistrationDebug *homography_debug,
                                        const cv::Point2f &shift);

    cv::Mat draw_tracks(const cv::Mat &frame,
                        const std::map<int, KalmanIoUTracker::Entity> &tracks) const;

    void write_debug_images(const cv::Mat &frame,
                            int frame_idx,
                            const cv::Mat &mask,
                            const cv::Mat &cc_mask,
                            const std::vector<cv::Rect> &detections) const;

    void process_image_outputs(const cv::Mat &registered_frame,
                               int frame_idx,
                               int frame_count,
                               const std::set<int> &ref_events,
                               const HomographyRegistrationDebug *homography_debug,
                               const cv::Point2f &shift,
                               const std::map<int, KalmanIoUTracker::Entity> &tracks,
                               const cv::Mat &mask,
                               const cv::Mat &cc_mask,
                               const std::vector<cv::Rect> &detections);
    void write_handshake(int frame_idx,
                         int frame_count,
                         const cv::Point2f &shift,
                         const std::map<int, KalmanIoUTracker::Entity> &tracks);

    void build_frame_vs_mask_video(float fps);

    void print_run_summary(const TimingSummary &reg_summary,
                           const TimingSummary &det_summary,
                           const TimingSummary &tracker_summary,
                           double loop_total,
                           float fps);

    static std::string pad_index(int idx);
    static void append_registrator_summary(std::vector<std::string> &lines, const TimingSummary &summary);
    static void append_detector_summary(std::vector<std::string> &lines, const TimingSummary &summary);
    static void append_tracker_summary(std::vector<std::string> &lines, const TimingSummary &summary);
    static bool has_value(const TimingSummary &summary, const std::string &key);
    static std::string to_string(const TimingSummary &summary, const std::string &key);
    static void append_value(std::vector<std::string> &lines,
                             const TimingSummary &summary,
                             const std::string &key,
                             const std::string &label);
};

int compute_total_frames(const DataStreamer &streamer);

}  // namespace cpp_utils
