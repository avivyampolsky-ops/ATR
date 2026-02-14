#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define ATR_VMD_STANDALONE
#include "cpp_modules/hpp/register_detect.hpp"
#include "cpp_modules/hpp/kalman.hpp"
#include "cpp_utils/hpp/cpp_utils.hpp"

namespace fs = std::filesystem;

using cpp_utils::Config;
using cpp_utils::DataStreamer;
using cpp_utils::DebugUtilsCpp;
using cpp_utils::compute_total_frames;

namespace {

float get_area_gate_min_scale(const Config &config) {
    float min_scale = config.get_float(
        "tracker.kalman.area_gate_min_scale__for_match",
        config.get_float("tracker.kalman.area_gate_min_scale", 0.5f)
    );
    return min_scale;
}

float get_area_gate_max_scale(const Config &config) {
    float max_scale = config.get_float(
        "tracker.kalman.area_gate_max_scale__for_match",
        config.get_float("tracker.kalman.area_gate_max_scale", 2.0f)
    );
    return max_scale;
}

TimingSummary make_reg_summary(const TimingSummary &combined) {
    TimingSummary out;
    out.enabled = combined.enabled;
    out.calls = combined.calls;
    auto copy_key = [&](const char *key) {
        auto it = combined.values.find(key);
        if (it != combined.values.end()) {
            out.values[key] = it->second;
        }
    };
    copy_key("registration_ms_avg");
    copy_key("gray_ms_avg");
    copy_key("keypoints_ms_avg");
    copy_key("match_ms_avg");
    copy_key("homography_ms_avg");
    copy_key("warp_ms_avg");
    copy_key("upload_ms_avg");
    copy_key("re_registration_calls");
    if (out.values.find("registration_ms_avg") != out.values.end()) {
        out.values["registration"] = out.values["registration_ms_avg"];
        out.values["registration_sum_ms_avg"] = out.values["registration_ms_avg"];
    }
    return out;
}

TimingSummary make_det_summary(const TimingSummary &combined) {
    TimingSummary out;
    out.enabled = combined.enabled;
    out.calls = combined.calls;
    auto copy_key = [&](const char *key) {
        auto it = combined.values.find(key);
        if (it != combined.values.end()) {
            out.values[key] = it->second;
        }
    };
    copy_key("blur_ms_avg");
    copy_key("diff_ms_avg");
    copy_key("bgsub_ms_avg");
    copy_key("contours_ms_avg");
    copy_key("bbox_creation_ms_avg");
    copy_key("download_ms_avg");
    if (combined.values.find("total_ms_avg") != combined.values.end() &&
        combined.values.find("registration_ms_avg") != combined.values.end()) {
        double det = combined.values.at("total_ms_avg") - combined.values.at("registration_ms_avg");
        out.values["detection_sum_ms_avg"] = std::max(0.0, det);
    } else if (combined.values.find("total_ms_avg") != combined.values.end()) {
        out.values["total_ms_avg"] = combined.values.at("total_ms_avg");
    }
    return out;
}

struct Args {
    std::string data_path;
    std::string out_dir;
    std::string config_path;
};

Args parse_args(int argc, char **argv, const std::string &default_config) {
    Args args;
    args.config_path = default_config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data-path" && i + 1 < argc) {
            args.data_path = argv[++i];
        } else if (arg == "--out-dir" && i + 1 < argc) {
            args.out_dir = argv[++i];
        } else if (arg == "--config-path" && i + 1 < argc) {
            args.config_path = argv[++i];
        }
    }
    return args;
}

}  // namespace

int main(int argc, char **argv) {
    fs::path config_default = fs::path(__FILE__).parent_path() / "config.yaml";
    Args args = parse_args(argc, argv, config_default.string());

    Config config;
    if (!config.load(args.config_path)) {
        return 1;
    }

    float fps_override = config.get_float("debug.fps", 30.0f);

    std::string data_path = args.data_path;
    if (data_path.empty()) {
        data_path = config.get_string("general.data_path", "");
    }
    std::string out_dir = args.out_dir;
    if (out_dir.empty()) {
        out_dir = config.get_string("general.out_dir", "");
    }
    if (data_path.empty() || out_dir.empty()) {
        std::cerr << "Missing data_path or out_dir. Set general.data_path/out_dir in config.yaml or pass --data-path/--out-dir.\n";
        return 1;
    }

    DataStreamer data_streamer;
    data_streamer.data_path = data_path;
    data_streamer.start_frame = std::max(0, config.get_int("debug.start_frame", 0));
    int end_frame = config.get_int("debug.end_frame", -1);
    if (end_frame >= 0) {
        data_streamer.end_frame = end_frame;
    }
    data_streamer.input_is_gray = config.get_bool("general.input_is_gray", false);
    data_streamer.fps_override = fps_override;

    if (!data_streamer.init_source()) {
        return 1;
    }

    float reference_window_ms = config.get_float("registration.reference_window_ms", 500.0f);
    int reference_window_frames = std::max(1, static_cast<int>(std::lround(data_streamer.fps * (reference_window_ms / 1000.0f))));
    config.set_runtime("registration.reference_window_frames", reference_window_frames);
    config.set_runtime("detection.learning_rate", std::min(1.0f, 1.0f / reference_window_frames));

    DebugUtilsCpp debug_utils(
        config,
        data_path,
        out_dir,
        data_streamer.fps,
        data_streamer.resolved_input_is_gray
    );
    debug_utils.set_stream_info(data_streamer);

    bool use_cpp = config.get_bool("general.use_cpp", true);
    bool use_cuda = config.get_bool("general.use_cuda", false);
    if (!use_cpp || !use_cuda) {
        std::cerr << "C++ main requires general.use_cpp=true and general.use_cuda=true.\n";
        return 1;
    }

    std::string registration_mode = config.get_string("registration.mode", "translation");
    std::string feature_type = config.get_string("registration.homography.feature_extractor", "FAST_BRIEF");
    std::string matcher_type = config.get_string("registration.homography.matcher", "BF");

    float downscale_factor = config.get_float("registration.downscale_factor", 1.0f);
    float detect_scale = config.get_float("detection.detect_scale", 1.0f);
    float learning_rate = config.get_float("detection.learning_rate", 0.05f);
    float mog2_var_threshold = config.get_float("detection.mog2_var_threshold", 20.0f);

    float knn_ratio = config.get_float("registration.homography.knn_ratio", 0.75f);
    float ransac_reproj_threshold = config.get_float("registration.homography.ransac_reproj_threshold", 5.0f);
    int min_inliers = config.get_int("registration.homography.min_inliers", 0);

    std::string translation_method = config.get_string("registration.translation.method", "phase");
    float phase_response_threshold = config.get_float("registration.translation.phase_response_threshold", 0.1f);
    bool phase_use_cached_fft = config.get_bool("registration.translation.phase_use_cached_fft", true);
    float max_shift = config.get_float("registration.translation.max_shift", 0.0f);

    bool debug_mode = config.get_bool("debug.debug_mode", true);
    bool enable_timing = debug_mode && config.get_bool("debug.enable_timing", true);
    bool draw_output = config.get_bool("debug.draw_output", false);
    bool show_pipeline = config.get_bool("debug.show_pipeline", false);

    if (translation_method != "phase" && registration_mode == "translation") {
        std::cerr << "C++ main supports only translation.method=phase\n";
    }

    if (show_pipeline) {
        std::cout << "\nPipeline:\n";
        std::cout << "  register+detect: " << registration_mode << " | cpp | gpu\n";
        std::cout << "  tracker: kalman | cpp | cpu\n\n";
    }

    std::unique_ptr<RegisterAndDetectTranslationGPU> translation_pipeline;
    std::unique_ptr<RegisterAndDetectHomographyGPU> homography_pipeline;
    if (registration_mode == "translation") {
        if (!RegisterAndDetectTranslationGPU::is_available()) {
            std::cerr << "CUDA translation pipeline unavailable.\n";
            return 1;
        }
        translation_pipeline = std::make_unique<RegisterAndDetectTranslationGPU>(
            data_streamer.ref,
            data_streamer.resolved_input_is_gray,
            downscale_factor,
            detect_scale,
            learning_rate,
            reference_window_frames,
            mog2_var_threshold,
            config.get_int("detection.min_contour_area", 50),
            config.get_int("detection.max_contour_area", 1500),
            config.get_float("detection.border_gate_percent", 1.0f),
            phase_response_threshold,
            0,
            debug_utils.reanchor_log_path,
            0.0f,
            max_shift,
            phase_use_cached_fft,
            enable_timing,
            draw_output
        );
    } else if (registration_mode == "homography") {
        if (!RegisterAndDetectHomographyGPU::is_available()) {
            std::cerr << "CUDA homography pipeline unavailable.\n";
            return 1;
        }
        homography_pipeline = std::make_unique<RegisterAndDetectHomographyGPU>(
            data_streamer.ref,
            data_streamer.resolved_input_is_gray,
            downscale_factor,
            detect_scale,
            learning_rate,
            reference_window_frames,
            mog2_var_threshold,
            config.get_int("detection.min_contour_area", 50),
            config.get_int("detection.max_contour_area", 1500),
            config.get_float("detection.border_gate_percent", 1.0f),
            feature_type,
            matcher_type,
            knn_ratio,
            ransac_reproj_threshold,
            min_inliers,
            0,
            0.0f,
            enable_timing,
            draw_output
        );
    } else {
        std::cerr << "Unsupported registration mode: " << registration_mode << "\n";
        return 1;
    }

    KalmanIoUTracker tracker(
        config.get_float("tracker.kalman.iou_thresh", 0.05f),
        config.get_int("tracker.kalman.max_lost", 8),
        config.get_float("tracker.kalman.min_move", 2.0f),
        config.get_float("tracker.kalman.ema_alpha", 0.3f),
        config.get_float("tracker.kalman.dist_gate_scale", 2.0f),
        config.get_float("tracker.kalman.easy_iou_thresh", 0.6f),
        get_area_gate_min_scale(config),
        get_area_gate_max_scale(config)
    );

    int frame_idx = data_streamer.start_frame;
    int frame_count = 0;
    double loop_total_time = 0.0;
    std::optional<cv::Point2f> prev_shift;
    cv::Point2f last_shift(0.0f, 0.0f);
    std::vector<cv::Rect> last_detections;
    cv::Mat last_registered_frame;

    TimingSummary tracker_summary;
    tracker_summary.enabled = enable_timing;
    tracker_summary.calls = 0;
    tracker_summary.values["update_ms_avg"] = 0.0;
    double tracker_update_total = 0.0;

    debug_utils.set_shape(cv::Size(data_streamer.ref.cols, data_streamer.ref.rows));

    std::vector<int> reference_events;
    std::set<int> reference_event_set;
    const int total_frames = compute_total_frames(data_streamer);
    int progress_last_print = -1;

    while (true) {
        cv::Mat frame = data_streamer.get_frame(frame_idx);
        if (frame.empty()) {
            break;
        }

        auto loop_start = std::chrono::high_resolution_clock::now();

        cv::Mat registered;
        if (translation_pipeline) {
            last_detections = translation_pipeline->register_and_detect(frame, &registered);
        } else {
            last_detections = homography_pipeline->register_and_detect(frame, &registered);
        }

        if (draw_output && !registered.empty()) {
            last_registered_frame = registered;
        } else {
            last_registered_frame = frame;
        }

        if (translation_pipeline) {
            auto shift = translation_pipeline->get_last_shift();
            last_shift.x = shift.first;
            last_shift.y = shift.second;
        } else {
            last_shift = cv::Point2f(0.0f, 0.0f);
        }

        reference_events = translation_pipeline ?
            translation_pipeline->get_reference_events() :
            homography_pipeline->get_reference_events();
        reference_event_set.clear();
        for (int ev : reference_events) {
            reference_event_set.insert(ev);
        }

        if (!reference_events.empty()) {
            int current_reg_frame = frame_count;
            if (reference_events.back() == current_reg_frame) {
                tracker.reset();
                prev_shift.reset();
            }
        }

        if (prev_shift.has_value()) {
            float dx = last_shift.x - prev_shift->x;
            float dy = last_shift.y - prev_shift->y;
            tracker.apply_shift(dx, dy);
        }
        prev_shift = last_shift;

        auto tracker_start = std::chrono::high_resolution_clock::now();
        std::map<int, KalmanIoUTracker::Entity> tracks = tracker.update(last_detections);
        auto tracker_end = std::chrono::high_resolution_clock::now();

        if (enable_timing) {
            double elapsed = std::chrono::duration<double>(tracker_end - tracker_start).count();
            tracker_summary.calls += 1;
            tracker_update_total += elapsed;
            tracker_summary.values["update_ms_avg"] = (tracker_update_total / tracker_summary.calls) * 1000.0;
        }

        auto loop_end = std::chrono::high_resolution_clock::now();
        loop_total_time += std::chrono::duration<double>(loop_end - loop_start).count();

        debug_utils.note_loop_call();

        HomographyRegistrationDebug homography_debug;
        const HomographyRegistrationDebug *homography_debug_ptr = nullptr;
        if (registration_mode == "homography") {
            homography_debug = homography_pipeline->get_last_registration_debug();
            homography_debug_ptr = &homography_debug;
            debug_utils.write_homography_log(frame_count, homography_debug_ptr);
        }

        cv::Mat mask;
        cv::Mat cc_mask;
        if (translation_pipeline) {
            mask = translation_pipeline->get_last_mask();
            cc_mask = translation_pipeline->get_last_cc_mask();
        } else {
            mask = homography_pipeline->get_last_mask();
            cc_mask = homography_pipeline->get_last_cc_mask();
        }
        debug_utils.process_image_outputs(
            last_registered_frame,
            frame_idx,
            frame_count,
            reference_event_set,
            homography_debug_ptr,
            last_shift,
            tracks,
            mask,
            cc_mask,
            last_detections
        );

        if (debug_mode) {
            if (total_frames > 0) {
                if (frame_count == 0 || frame_count - progress_last_print >= 10) {
                    progress_last_print = frame_count;
                    double pct = (static_cast<double>(frame_count + 1) / total_frames) * 100.0;
                    std::cout << "\rProcessed " << (frame_count + 1) << "/" << total_frames
                              << " (" << std::fixed << std::setprecision(1) << pct << "%)"
                              << std::flush;
                }
            } else {
                if (frame_count == 0 || frame_count - progress_last_print >= 10) {
                    progress_last_print = frame_count;
                    std::cout << "\rProcessed " << (frame_count + 1) << " frames" << std::flush;
                }
            }
        }

        frame_idx += 1;
        frame_count += 1;
    }

    if (debug_mode && progress_last_print >= 0) {
        std::cout << "\n";
    }

    debug_utils.close();
    data_streamer.close();

    debug_utils.write_reference_events_graph(reference_events);

    TimingSummary combined_summary;
    if (translation_pipeline) {
        combined_summary = translation_pipeline->timing_summary();
    } else {
        combined_summary = homography_pipeline->timing_summary();
    }
    TimingSummary reg_summary = make_reg_summary(combined_summary);
    TimingSummary det_summary = make_det_summary(combined_summary);

    debug_utils.print_run_summary(reg_summary, det_summary, tracker_summary, loop_total_time, data_streamer.fps);

    return 0;
}
