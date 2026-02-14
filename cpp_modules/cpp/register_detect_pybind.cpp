#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <optional>
#include <vector>

namespace py = pybind11;

// Core headers
#include "../hpp/register_detect.hpp"
#include "../hpp/kalman.hpp"
#include "../../cpp_utils/hpp/cpp_utils.hpp"

namespace {

cv::Mat array_to_mat(const py::array &array) {
    py::array arr = py::array::ensure(array, py::array::c_style | py::array::forcecast);
    if (!arr) {
        throw std::runtime_error("Expected a contiguous numpy array.");
    }
    py::buffer_info info = arr.request();
    if (info.ndim != 2 && info.ndim != 3) {
        throw std::runtime_error("Expected a 2D or 3D numpy array.");
    }
    if (info.format != py::format_descriptor<uint8_t>::format()) {
        throw std::runtime_error("Expected uint8 numpy array.");
    }

    int height = static_cast<int>(info.shape[0]);
    int width = static_cast<int>(info.shape[1]);
    if (info.ndim == 2) {
        cv::Mat mat(height, width, CV_8UC1);
        std::memcpy(mat.data, info.ptr, mat.total() * mat.elemSize());
        return mat;
    }
    int channels = static_cast<int>(info.shape[2]);
    if (channels != 3) {
        throw std::runtime_error("Expected 3-channel BGR numpy array.");
    }
    cv::Mat mat(height, width, CV_8UC3);
    std::memcpy(mat.data, info.ptr, mat.total() * mat.elemSize());
    return mat;
}

py::array mat_to_array(const cv::Mat &mat) {
    if (mat.empty()) {
        return py::array();
    }
    cv::Mat contiguous = mat;
    if (!mat.isContinuous()) {
        contiguous = mat.clone();
    }
    const int channels = contiguous.channels();
    std::vector<py::ssize_t> shape;
    if (channels == 1) {
        shape = {contiguous.rows, contiguous.cols};
    } else {
        shape = {contiguous.rows, contiguous.cols, channels};
    }
    py::array out(py::dtype::of<uint8_t>(), shape);
    std::memcpy(out.mutable_data(), contiguous.data, contiguous.total() * contiguous.elemSize());
    return out;
}

py::array detections_to_array(const std::vector<cv::Rect> &detections) {
    py::array out(py::dtype::of<int32_t>(),
                  {static_cast<py::ssize_t>(detections.size()), static_cast<py::ssize_t>(4)});
    auto buf = out.mutable_unchecked<int32_t, 2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(detections.size()); ++i) {
        const cv::Rect &r = detections[static_cast<size_t>(i)];
        buf(i, 0) = r.x;
        buf(i, 1) = r.y;
        buf(i, 2) = r.width;
        buf(i, 3) = r.height;
    }
    return out;
}

std::vector<cv::Rect> detections_from_array(const py::array &observations) {
    auto arr = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(observations);
    if (!arr) {
        throw std::runtime_error("Expected a contiguous numpy array.");
    }
    py::buffer_info info = arr.request();
    if (info.ndim != 2 || info.shape[1] != 4) {
        throw std::runtime_error("Expected Nx4 observations array.");
    }
    const int count = static_cast<int>(info.shape[0]);
    const auto *ptr = static_cast<const float *>(info.ptr);
    std::vector<cv::Rect> out;
    out.reserve(count);
    for (int i = 0; i < count; ++i) {
        int x = static_cast<int>(std::round(ptr[i * 4 + 0]));
        int y = static_cast<int>(std::round(ptr[i * 4 + 1]));
        int w = static_cast<int>(std::round(ptr[i * 4 + 2]));
        int h = static_cast<int>(std::round(ptr[i * 4 + 3]));
        out.emplace_back(x, y, w, h);
    }
    return out;
}

py::dict timing_to_dict(const TimingSummary &summary) {
    py::dict out;
    out["enabled"] = summary.enabled;
    out["calls"] = summary.calls;
    for (const auto &kv : summary.values) {
        out[py::str(kv.first)] = kv.second;
    }
    return out;
}

py::dict translation_debug_to_dict(const TranslationRegistrationDebug &debug) {
    py::dict out;
    out["phase_response_threshold"] = debug.phase_response_threshold;
    out["response"] = debug.response;
    out["phase_ok"] = debug.phase_ok;
    out["dx"] = debug.dx;
    out["dy"] = debug.dy;
    out["reanchor_reason"] = debug.reanchor_reason;
    return out;
}

py::dict homography_debug_to_dict(const HomographyRegistrationDebug &debug) {
    py::dict out;
    out["fail_reason"] = debug.fail_reason;
    out["match_count"] = debug.match_count;
    out["inlier_count"] = debug.inlier_count;
    out["overlap_ratio"] = debug.overlap_ratio;
    out["reference_reason"] = debug.reference_reason;
    return out;
}

py::array ref_events_to_array(const std::vector<int> &events) {
    py::array_t<int32_t> out(events.size());
    auto buf = out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(events.size()); ++i) {
        buf(i) = events[static_cast<size_t>(i)];
    }
    return out;
}

template <typename T>
T get_cfg(const py::dict &cfg, const char *key, const T &fallback) {
    if (cfg.contains(py::str(key))) {
        return cfg[py::str(key)].cast<T>();
    }
    return fallback;
}

template <typename T>
T get_cfg(const cpp_utils::Config &cfg, const char *key, const T &fallback);

template <>
bool get_cfg<bool>(const cpp_utils::Config &cfg, const char *key, const bool &fallback) {
    return cfg.get_bool(key, fallback);
}

template <>
int get_cfg<int>(const cpp_utils::Config &cfg, const char *key, const int &fallback) {
    return cfg.get_int(key, fallback);
}

template <>
float get_cfg<float>(const cpp_utils::Config &cfg, const char *key, const float &fallback) {
    return cfg.get_float(key, fallback);
}

template <>
std::string get_cfg<std::string>(const cpp_utils::Config &cfg, const char *key, const std::string &fallback) {
    return cfg.get_string(key, fallback);
}

class ATRVMDTrackerCpp {
public:
    ATRVMDTrackerCpp(const py::array &reference, const py::dict &config)
        : input_is_gray(get_cfg<bool>(config, "general.input_is_gray", false)),
          registration_mode(get_cfg<std::string>(config, "registration.mode", "translation")),
          enable_timing(get_cfg<bool>(config, "debug.enable_timing", true)) {
        init(reference,
             get_cfg<bool>(config, "general.use_cuda", true),
             get_cfg<float>(config, "registration.downscale_factor", 1.0f),
             get_cfg<float>(config, "detection.detect_scale", 1.0f),
             get_cfg<float>(config, "detection.learning_rate", 0.05f),
             get_reference_window_frames(config),
             get_cfg<float>(config, "detection.mog2_var_threshold", 20.0f),
             get_cfg<int>(config, "detection.min_contour_area", 50),
             get_cfg<int>(config, "detection.max_contour_area", 1500),
             get_cfg<float>(config, "detection.border_gate_percent", 1.0f),
             get_cfg<int>(config, "registration.reanchor_boost_frames", 0),
             get_cfg<float>(config, "registration.reanchor_boost_lr", 0.0f),
             get_cfg<float>(config, "debug.fps", 30.0f),
             get_cfg<std::string>(config, "debug.reanchor_log_path", ""),
             get_cfg<float>(config, "registration.translation.phase_response_threshold", 0.1f),
             get_cfg<float>(config, "registration.translation.max_shift", 0.0f),
             get_cfg<bool>(config, "registration.translation.phase_use_cached_fft", true),
             get_cfg<std::string>(config, "registration.homography.feature_extractor", "ORB"),
             get_cfg<std::string>(config, "registration.homography.matcher", "BF"),
             get_cfg<float>(config, "registration.homography.knn_ratio", 0.75f),
             get_cfg<float>(config, "registration.homography.ransac_reproj_threshold", 5.0f),
             get_cfg<int>(config, "registration.homography.min_inliers", 0),
             get_cfg<float>(config, "tracker.kalman.iou_thresh", 0.05f),
             get_cfg<int>(config, "tracker.kalman.max_lost", 8),
             get_cfg<float>(config, "tracker.kalman.min_move", 2.0f),
             get_cfg<float>(config, "tracker.kalman.ema_alpha", 0.3f),
             get_cfg<float>(config, "tracker.kalman.dist_gate_scale", 2.0f),
             get_cfg<float>(config, "tracker.kalman.easy_iou_thresh", 0.6f),
             get_cfg<float>(
                 config,
                 "tracker.kalman.area_gate_min_scale__for_match",
                 get_cfg<float>(config, "tracker.kalman.area_gate_min_scale", 0.5f)
             ),
             get_cfg<float>(
                 config,
                 "tracker.kalman.area_gate_max_scale__for_match",
                 get_cfg<float>(config, "tracker.kalman.area_gate_max_scale", 2.0f)
             ));
    }

    ATRVMDTrackerCpp(const py::array &reference, const std::string &config_path = "")
        : input_is_gray(false),
          registration_mode("translation"),
          enable_timing(true) {
        cpp_utils::Config cfg;
        std::string path = config_path.empty()
            ? (std::filesystem::current_path() / "config.yaml").string()
            : config_path;
        if (!cfg.load(path)) {
            throw std::runtime_error("Failed to load config.yaml: " + path);
        }
        input_is_gray = get_cfg<bool>(cfg, "general.input_is_gray", false);
        registration_mode = get_cfg<std::string>(cfg, "registration.mode", "translation");
        enable_timing = get_cfg<bool>(cfg, "debug.enable_timing", true);

        init(reference,
             get_cfg<bool>(cfg, "general.use_cuda", true),
             get_cfg<float>(cfg, "registration.downscale_factor", 1.0f),
             get_cfg<float>(cfg, "detection.detect_scale", 1.0f),
             get_cfg<float>(cfg, "detection.learning_rate", 0.05f),
             get_reference_window_frames(cfg),
             get_cfg<float>(cfg, "detection.mog2_var_threshold", 20.0f),
             get_cfg<int>(cfg, "detection.min_contour_area", 50),
             get_cfg<int>(cfg, "detection.max_contour_area", 1500),
             get_cfg<float>(cfg, "detection.border_gate_percent", 1.0f),
             get_cfg<int>(cfg, "registration.reanchor_boost_frames", 0),
             get_cfg<float>(cfg, "registration.reanchor_boost_lr", 0.0f),
             get_cfg<float>(cfg, "debug.fps", 30.0f),
             get_cfg<std::string>(cfg, "debug.reanchor_log_path", ""),
             get_cfg<float>(cfg, "registration.translation.phase_response_threshold", 0.1f),
             get_cfg<float>(cfg, "registration.translation.max_shift", 0.0f),
             get_cfg<bool>(cfg, "registration.translation.phase_use_cached_fft", true),
             get_cfg<std::string>(cfg, "registration.homography.feature_extractor", "ORB"),
             get_cfg<std::string>(cfg, "registration.homography.matcher", "BF"),
             get_cfg<float>(cfg, "registration.homography.knn_ratio", 0.75f),
             get_cfg<float>(cfg, "registration.homography.ransac_reproj_threshold", 5.0f),
             get_cfg<int>(cfg, "registration.homography.min_inliers", 0),
             get_cfg<float>(cfg, "tracker.kalman.iou_thresh", 0.05f),
             get_cfg<int>(cfg, "tracker.kalman.max_lost", 8),
             get_cfg<float>(cfg, "tracker.kalman.min_move", 2.0f),
             get_cfg<float>(cfg, "tracker.kalman.ema_alpha", 0.3f),
             get_cfg<float>(cfg, "tracker.kalman.dist_gate_scale", 2.0f),
             get_cfg<float>(cfg, "tracker.kalman.easy_iou_thresh", 0.6f),
             get_cfg<float>(
                 cfg,
                 "tracker.kalman.area_gate_min_scale__for_match",
                 get_cfg<float>(cfg, "tracker.kalman.area_gate_min_scale", 0.5f)
             ),
             get_cfg<float>(
                 cfg,
                 "tracker.kalman.area_gate_max_scale__for_match",
                 get_cfg<float>(cfg, "tracker.kalman.area_gate_max_scale", 2.0f)
             ));
    }

    py::dict process_frame(const py::array &frame) {
        cv::Mat input = array_to_mat(frame);
        cv::Mat registered;
        std::vector<cv::Rect> dets;
        if (translation) {
            dets = translation->register_and_detect(input, &registered);
            auto shift = translation->get_last_shift();
            last_shift = cv::Point2f(shift.first, shift.second);
            reference_events = translation->get_reference_events();
        } else {
            dets = homography->register_and_detect(input, &registered);
            auto shift = homography->get_last_shift();
            last_shift = cv::Point2f(shift.first, shift.second);
            reference_events = homography->get_reference_events();
        }

        if (!reference_events.empty()) {
            if (reference_events.back() == frame_count) {
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
        last_detections = dets;
        last_registered = registered.empty() ? input : registered;

        std::map<int, KalmanIoUTracker::Entity> tracks = tracker.update(dets);
        frame_count += 1;
        return py::cast(tracks);
    }

    py::array get_last_registered_frame() const {
        return mat_to_array(last_registered);
    }

    py::array get_last_mask() const {
        if (translation) {
            return mat_to_array(translation->get_last_mask());
        }
        return mat_to_array(homography->get_last_mask());
    }

    py::array get_last_cc_mask() const {
        if (translation) {
            return mat_to_array(translation->get_last_cc_mask());
        }
        return mat_to_array(homography->get_last_cc_mask());
    }

    py::array get_last_bb_mask() const {
        if (translation) {
            return mat_to_array(translation->get_last_bb_mask());
        }
        return mat_to_array(homography->get_last_bb_mask());
    }

    py::array get_reference_events() const {
        return ref_events_to_array(reference_events);
    }

    py::tuple get_last_shift() const {
        return py::make_tuple(last_shift.x, last_shift.y);
    }

    py::array get_last_detections() const {
        return detections_to_array(last_detections);
    }

private:
    int get_reference_window_frames(const py::dict &config) const {
        int reference_window_frames = get_cfg<int>(config, "registration.reference_window_frames", -1);
        if (reference_window_frames <= 0) {
            float reference_window_ms = get_cfg<float>(config, "registration.reference_window_ms", 500.0f);
            float fps = get_cfg<float>(config, "debug.fps", 30.0f);
            reference_window_frames = std::max(1, static_cast<int>(std::lround(fps * (reference_window_ms / 1000.0f))));
        }
        return reference_window_frames;
    }

    int get_reference_window_frames(const cpp_utils::Config &config) const {
        int reference_window_frames = get_cfg<int>(config, "registration.reference_window_frames", -1);
        if (reference_window_frames <= 0) {
            float reference_window_ms = get_cfg<float>(config, "registration.reference_window_ms", 500.0f);
            float fps = get_cfg<float>(config, "debug.fps", 30.0f);
            reference_window_frames = std::max(1, static_cast<int>(std::lround(fps * (reference_window_ms / 1000.0f))));
        }
        return reference_window_frames;
    }

    void init(const py::array &reference,
              bool use_cuda,
              float downscale_factor,
              float detect_scale,
              float learning_rate,
              int reference_window_frames,
              float mog2_var_threshold,
              int min_contour_area,
              int max_contour_area,
              float border_gate_percent,
              int reanchor_boost_frames,
              float reanchor_boost_lr,
              float fps,
              const std::string &reanchor_log_path,
              float phase_response_threshold,
              float max_shift,
              bool phase_use_cached_fft,
              const std::string &feature_type,
              const std::string &matcher_type,
              float knn_ratio,
              float ransac_reproj_threshold,
              int min_inliers,
              float iou_thresh,
              int max_lost,
              float min_move,
              float ema_alpha,
              float dist_gate_scale,
              float easy_iou_thresh,
              float area_gate_min_scale,
              float area_gate_max_scale) {
        if (!use_cuda) {
            throw std::runtime_error("ATRVMDTrackerCpp supports only CUDA (general.use_cuda=true).");
        }
        if (reference_window_frames <= 0) {
            reference_window_frames = std::max(1, static_cast<int>(std::lround(fps * (500.0f / 1000.0f))));
        }
        if (registration_mode == "translation") {
            translation = std::make_unique<RegisterAndDetectTranslationGPU>(
                array_to_mat(reference),
                input_is_gray,
                downscale_factor,
                detect_scale,
                learning_rate,
                reference_window_frames,
                mog2_var_threshold,
                min_contour_area,
                max_contour_area,
                border_gate_percent,
                phase_response_threshold,
                reanchor_boost_frames,
                reanchor_log_path,
                reanchor_boost_lr,
                max_shift,
                phase_use_cached_fft,
                enable_timing,
                true
            );
        } else if (registration_mode == "homography") {
            homography = std::make_unique<RegisterAndDetectHomographyGPU>(
                array_to_mat(reference),
                input_is_gray,
                downscale_factor,
                detect_scale,
                learning_rate,
                reference_window_frames,
                mog2_var_threshold,
                min_contour_area,
                max_contour_area,
                border_gate_percent,
                feature_type,
                matcher_type,
                knn_ratio,
                ransac_reproj_threshold,
                min_inliers,
                reanchor_boost_frames,
                reanchor_boost_lr,
                enable_timing,
                true
            );
        } else {
            throw std::runtime_error("Unsupported registration.mode (expected 'translation' or 'homography').");
        }

        tracker = KalmanIoUTracker(
            iou_thresh,
            max_lost,
            min_move,
            ema_alpha,
            dist_gate_scale,
            easy_iou_thresh,
            area_gate_min_scale,
            area_gate_max_scale
        );
    }

    bool input_is_gray = false;
    std::string registration_mode;
    bool enable_timing = true;

    std::unique_ptr<RegisterAndDetectTranslationGPU> translation;
    std::unique_ptr<RegisterAndDetectHomographyGPU> homography;
    KalmanIoUTracker tracker = KalmanIoUTracker();

    std::optional<cv::Point2f> prev_shift;
    cv::Point2f last_shift = cv::Point2f(0.0f, 0.0f);
    std::vector<int> reference_events;
    std::vector<cv::Rect> last_detections;
    cv::Mat last_registered;
    int frame_count = 0;
};

}  // namespace

PYBIND11_MODULE(_register_detect_cpp, m) {
    m.doc() = "GPU register+detect pipeline (C++/pybind11).";

    py::class_<RegisterAndDetectTranslationGPU>(m, "RegisterAndDetectTranslationGPUCpp")
        .def(py::init([](
                const py::array &reference,
                bool input_is_gray,
                float downscale_factor,
                float detect_scale,
                float learning_rate,
                int reference_window_frames,
                float mog2_var_threshold,
                int min_contour_area,
                int max_contour_area,
                float border_gate_percent,
                float phase_response_threshold,
                int reanchor_boost_frames,
                const std::string &reanchor_log_path,
                float reanchor_boost_lr,
                float max_shift,
                bool phase_use_cached_fft,
                bool enable_timing,
                bool return_registered) {
            return new RegisterAndDetectTranslationGPU(
                array_to_mat(reference),
                input_is_gray,
                downscale_factor,
                detect_scale,
                learning_rate,
                reference_window_frames,
                mog2_var_threshold,
                min_contour_area,
                max_contour_area,
                border_gate_percent,
                phase_response_threshold,
                reanchor_boost_frames,
                reanchor_log_path,
                reanchor_boost_lr,
                max_shift,
                phase_use_cached_fft,
                enable_timing,
                return_registered
            );
        }),
         py::arg("reference"),
         py::arg("input_is_gray"),
         py::arg("downscale_factor") = 1.0f,
         py::arg("detect_scale") = 1.0f,
         py::arg("learning_rate") = 0.05f,
         py::arg("reference_window_frames") = 1,
         py::arg("mog2_var_threshold") = 20.0f,
         py::arg("min_contour_area") = 50,
         py::arg("max_contour_area") = 1500,
         py::arg("border_gate_percent") = 1.0f,
         py::arg("phase_response_threshold") = 0.1f,
         py::arg("reanchor_boost_frames") = 0,
         py::arg("reanchor_log_path") = "",
         py::arg("reanchor_boost_lr") = 0.0f,
         py::arg("max_shift") = 0.0f,
         py::arg("phase_use_cached_fft") = true,
         py::arg("enable_timing") = true,
         py::arg("return_registered") = true)
        .def_static("is_available", &RegisterAndDetectTranslationGPU::is_available)
        .def("register_and_detect", [](RegisterAndDetectTranslationGPU &self, const py::array &frame) {
            cv::Mat input = array_to_mat(frame);
            cv::Mat registered;
            std::vector<cv::Rect> dets = self.register_and_detect(input, &registered);
            py::array det_arr = detections_to_array(dets);
            if (registered.empty()) {
                return py::make_tuple(det_arr, py::none());
            }
            return py::make_tuple(det_arr, mat_to_array(registered));
        })
        .def("timing_summary", [](RegisterAndDetectTranslationGPU &self, bool reset) {
            return timing_to_dict(self.timing_summary(reset));
        }, py::arg("reset") = false)
        .def("reset_timing", &RegisterAndDetectTranslationGPU::reset_timing)
        .def("get_last_mask", [](RegisterAndDetectTranslationGPU &self) {
            return mat_to_array(self.get_last_mask());
        })
        .def("get_last_cc_mask", [](RegisterAndDetectTranslationGPU &self) {
            return mat_to_array(self.get_last_cc_mask());
        })
        .def("get_last_bb_mask", [](RegisterAndDetectTranslationGPU &self) {
            return mat_to_array(self.get_last_bb_mask());
        })
        .def("get_reference_events", [](RegisterAndDetectTranslationGPU &self) {
            return ref_events_to_array(self.get_reference_events());
        })
        .def("get_last_shift", [](RegisterAndDetectTranslationGPU &self) {
            auto shift = self.get_last_shift();
            return py::make_tuple(shift.first, shift.second);
        })
        .def("get_last_registration_debug", [](RegisterAndDetectTranslationGPU &self) {
            return translation_debug_to_dict(self.get_last_registration_debug());
        });

    py::class_<RegisterAndDetectHomographyGPU>(m, "RegisterAndDetectHomographyGPUCpp")
        .def(py::init([](
                const py::array &reference,
                bool input_is_gray,
                float downscale_factor,
                float detect_scale,
                float learning_rate,
                int reference_window_frames,
                float mog2_var_threshold,
                int min_contour_area,
                int max_contour_area,
                float border_gate_percent,
                const std::string &feature_type,
                const std::string &matcher_type,
                float knn_ratio,
                float ransac_reproj_threshold,
                int min_inliers,
                int reanchor_boost_frames,
                float reanchor_boost_lr,
                bool enable_timing,
                bool return_registered) {
            return new RegisterAndDetectHomographyGPU(
                array_to_mat(reference),
                input_is_gray,
                downscale_factor,
                detect_scale,
                learning_rate,
                reference_window_frames,
                mog2_var_threshold,
                min_contour_area,
                max_contour_area,
                border_gate_percent,
                feature_type,
                matcher_type,
                knn_ratio,
                ransac_reproj_threshold,
                min_inliers,
                reanchor_boost_frames,
                reanchor_boost_lr,
                enable_timing,
                return_registered
            );
        }),
         py::arg("reference"),
         py::arg("input_is_gray"),
         py::arg("downscale_factor") = 1.0f,
         py::arg("detect_scale") = 1.0f,
         py::arg("learning_rate") = 0.05f,
         py::arg("reference_window_frames") = 1,
         py::arg("mog2_var_threshold") = 20.0f,
         py::arg("min_contour_area") = 50,
         py::arg("max_contour_area") = 1500,
         py::arg("border_gate_percent") = 1.0f,
         py::arg("feature_type") = "ORB",
         py::arg("matcher_type") = "BF",
         py::arg("knn_ratio") = 0.75f,
         py::arg("ransac_reproj_threshold") = 5.0f,
         py::arg("min_inliers") = 0,
         py::arg("reanchor_boost_frames") = 0,
         py::arg("reanchor_boost_lr") = 0.0f,
         py::arg("enable_timing") = true,
         py::arg("return_registered") = true)
        .def_static("is_available", &RegisterAndDetectHomographyGPU::is_available)
        .def("register_and_detect", [](RegisterAndDetectHomographyGPU &self, const py::array &frame) {
            cv::Mat input = array_to_mat(frame);
            cv::Mat registered;
            std::vector<cv::Rect> dets = self.register_and_detect(input, &registered);
            py::array det_arr = detections_to_array(dets);
            if (registered.empty()) {
                return py::make_tuple(det_arr, py::none());
            }
            return py::make_tuple(det_arr, mat_to_array(registered));
        })
        .def("timing_summary", [](RegisterAndDetectHomographyGPU &self, bool reset) {
            return timing_to_dict(self.timing_summary(reset));
        }, py::arg("reset") = false)
        .def("reset_timing", &RegisterAndDetectHomographyGPU::reset_timing)
        .def("get_last_mask", [](RegisterAndDetectHomographyGPU &self) {
            return mat_to_array(self.get_last_mask());
        })
        .def("get_last_cc_mask", [](RegisterAndDetectHomographyGPU &self) {
            return mat_to_array(self.get_last_cc_mask());
        })
        .def("get_last_bb_mask", [](RegisterAndDetectHomographyGPU &self) {
            return mat_to_array(self.get_last_bb_mask());
        })
        .def("get_reference_events", [](RegisterAndDetectHomographyGPU &self) {
            return ref_events_to_array(self.get_reference_events());
        })
        .def("get_last_shift", [](RegisterAndDetectHomographyGPU &self) {
            auto shift = self.get_last_shift();
            return py::make_tuple(shift.first, shift.second);
        })
        .def("get_last_registration_debug", [](RegisterAndDetectHomographyGPU &self) {
            return homography_debug_to_dict(self.get_last_registration_debug());
        });

    py::class_<KalmanIoUTracker::Entity>(m, "KalmanEntity")
        .def_property_readonly("id", [](const KalmanIoUTracker::Entity &e) { return e.id; })
        .def_property_readonly("bbox", [](const KalmanIoUTracker::Entity &e) {
            py::array_t<int32_t> out(4);
            auto buf = out.mutable_unchecked<1>();
            buf(0) = static_cast<int32_t>(std::round(e.bbox.x));
            buf(1) = static_cast<int32_t>(std::round(e.bbox.y));
            buf(2) = static_cast<int32_t>(std::round(e.bbox.w));
            buf(3) = static_cast<int32_t>(std::round(e.bbox.h));
            return out;
        })
        .def_property_readonly("age", [](const KalmanIoUTracker::Entity &e) { return e.age; })
        .def_property_readonly("lost", [](const KalmanIoUTracker::Entity &e) { return e.lost; })
        .def_property_readonly("moving", [](const KalmanIoUTracker::Entity &e) { return e.moving; })
        .def_property_readonly("ema_area", [](const KalmanIoUTracker::Entity &e) { return e.ema_area; });

    py::class_<KalmanIoUTracker>(m, "KalmanIoUTrackerCpp")
        .def(py::init<float, int, float, float, float, float, float, float>(),
             py::arg("iou_thresh") = 0.05f,
             py::arg("max_lost") = 8,
             py::arg("min_move") = 2.0f,
             py::arg("ema_alpha") = 0.3f,
             py::arg("dist_gate_scale") = 2.0f,
             py::arg("easy_iou_thresh") = 0.6f,
             py::arg("area_gate_min_scale") = 0.5f,
             py::arg("area_gate_max_scale") = 2.0f)
        .def("apply_shift", &KalmanIoUTracker::apply_shift)
        .def("reset", &KalmanIoUTracker::reset)
        .def("update", [](KalmanIoUTracker &self, const py::array &observations) {
            auto dets = detections_from_array(observations);
            return self.update(dets);
        })
        .def("get_tracks", &KalmanIoUTracker::get_tracks, py::return_value_policy::reference_internal);

    py::class_<ATRVMDTrackerCpp>(m, "ATRVMDTrackerCpp")
        .def(py::init<const py::array &>(),
             py::arg("reference"))
        .def(py::init<const py::array &, const std::string &>(),
             py::arg("reference"),
             py::arg("config_path") = "")
        .def(py::init<const py::array &, const py::dict &>(),
             py::arg("reference"),
             py::arg("config"))
        .def("process_frame", &ATRVMDTrackerCpp::process_frame)
        .def("get_last_registered_frame", &ATRVMDTrackerCpp::get_last_registered_frame)
        .def("get_last_mask", &ATRVMDTrackerCpp::get_last_mask)
        .def("get_last_cc_mask", &ATRVMDTrackerCpp::get_last_cc_mask)
        .def("get_last_bb_mask", &ATRVMDTrackerCpp::get_last_bb_mask)
        .def("get_reference_events", &ATRVMDTrackerCpp::get_reference_events)
        .def("get_last_shift", &ATRVMDTrackerCpp::get_last_shift)
        .def("get_last_detections", &ATRVMDTrackerCpp::get_last_detections);
}
