#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudawarping.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>
#include <utility>

struct TimingSummary {
    bool enabled = true;
    int calls = 0;
    std::map<std::string, double> values;
};

struct TranslationRegistrationDebug {
    float phase_response_threshold = 0.0f;
    float response = 0.0f;
    bool phase_ok = false;
    float dx = 0.0f;
    float dy = 0.0f;
    std::string reanchor_reason;
};

struct HomographyRegistrationDebug {
    std::string fail_reason;
    int match_count = 0;
    int inlier_count = 0;
    double overlap_ratio = 0.0;
    std::string reference_reason;
};

class RegisterAndDetectTranslationGPU {
public:
    RegisterAndDetectTranslationGPU(
        const cv::Mat &reference,
        bool input_is_gray,
        float downscale_factor = 1.0f,
        float detect_scale = 1.0f,
        float learning_rate = 0.05f,
        int reference_window_frames = 1,
        float mog2_var_threshold = 20.0f,
        int min_contour_area = 50,
        int max_contour_area = 1500,
        float border_gate_percent = 1.0f,
        float phase_response_threshold = 0.1f,
        int reanchor_boost_frames = 0,
        const std::string &reanchor_log_path = "",
        float reanchor_boost_lr = 0.0f,
        float max_shift = 0.0f,
        bool phase_use_cached_fft = true,
        bool enable_timing = true,
        bool return_registered = true
    )
        : input_is_gray_(input_is_gray),
          downscale_factor_(downscale_factor > 0.0f ? downscale_factor : 1.0f),
          detect_scale_(detect_scale > 0.0f ? detect_scale : 1.0f),
          scale_inv_(1.0f / detect_scale_),
          learning_rate_(learning_rate),
          reference_window_frames_(reference_window_frames > 0 ? reference_window_frames : 1),
          mog2_history_(reference_window_frames_),
          mog2_var_threshold_(mog2_var_threshold),
          min_contour_area_(std::max(0, min_contour_area)),
          max_contour_area_(std::max(min_contour_area_, max_contour_area)),
          border_gate_percent_(std::clamp(border_gate_percent, 0.0f, 49.0f)),
          phase_response_threshold_(phase_response_threshold),
          reanchor_boost_frames_(reanchor_boost_frames > 0 ? reanchor_boost_frames : 0),
          max_shift_(max_shift > 0.0f ? max_shift : 0.0f),
          reanchor_boost_lr_(reanchor_boost_lr > 0.0f ? reanchor_boost_lr : learning_rate_),
          phase_use_cached_fft_(phase_use_cached_fft),
          timing_enabled_(enable_timing),
          return_registered_(return_registered) {
        if (reference.empty()) {
            throw std::runtime_error("Expected non-empty reference frame.");
        }
        cv::Mat ref_mat = reference;
        set_geometry(ref_mat);
        set_reference_mat(ref_mat, false);
        init_detector_buffers();

        backsub_ = cv::cuda::createBackgroundSubtractorMOG2(
            mog2_history_,   // history length (frames)
            mog2_var_threshold,    // variance threshold for foreground
            false  // disable shadow detection for speed
        );

        int type = input_is_gray_ ? CV_8UC1 : CV_8UC3;
        gaussian_filter_ = cv::cuda::createGaussianFilter(
            type,
            type,
            cv::Size(3, 3),
            0.0
        );

        if (!reanchor_log_path.empty()) {
            reanchor_log_.open(reanchor_log_path, std::ios::out);
            if (reanchor_log_.is_open()) {
                reanchor_log_ << "frame,response,phase_ok,dx,dy,reason\n";
                reanchor_log_.flush();
            }
        }

        if (timing_enabled_) {
            timing_ = {
                {"calls", 0.0},
                {"registration", 0.0},
                {"blur", 0.0},
                {"bgsub", 0.0},
                {"contours", 0.0},
                {"upload", 0.0},
                {"download", 0.0},
                {"bbox_creation", 0.0},
                {"total", 0.0},
            };
        }
    }

    static bool is_available() {
        try {
            return cv::cuda::getCudaEnabledDeviceCount() > 0;
        } catch (const cv::Exception &) {
            return false;
        }
    }

    TimingSummary timing_summary(bool reset = false) {
        TimingSummary out;
        if (!timing_enabled_) {
            out.enabled = false;
            return out;
        }
        const double calls = timing_["calls"];
        out.enabled = true;
        out.calls = static_cast<int>(calls);
        if (calls > 0.0) {
            out.values["total_ms_avg"] = (timing_["total"] / calls) * 1000.0;
            out.values["registration_ms_avg"] = (timing_["registration"] / calls) * 1000.0;
            out.values["blur_ms_avg"] = (timing_["blur"] / calls) * 1000.0;
            out.values["bgsub_ms_avg"] = (timing_["bgsub"] / calls) * 1000.0;
            out.values["contours_ms_avg"] = (timing_["contours"] / calls) * 1000.0;
            out.values["upload_ms_avg"] = (timing_["upload"] / calls) * 1000.0;
            out.values["download_ms_avg"] = (timing_["download"] / calls) * 1000.0;
            out.values["bbox_creation_ms_avg"] = (timing_["bbox_creation"] / calls) * 1000.0;
        }
        if (reset) {
            reset_timing();
        }
        return out;
    }

    void reset_timing() {
        if (!timing_enabled_) {
            return;
        }
        for (auto &kv : timing_) {
            kv.second = 0.0;
        }
    }

    std::vector<cv::Rect> register_and_detect(const cv::Mat &frame, cv::Mat *registered_out = nullptr) {
        const auto start = now();
        if (frame.empty()) {
            throw std::runtime_error("Expected non-empty frame for GPU register/detect.");
        }
        cv::Mat frame_cpu = frame;
        if (input_is_gray_ && frame_cpu.channels() != 1) {
            throw std::runtime_error("Expected grayscale frame for GPU register/detect.");
        }
        if (!input_is_gray_ && frame_cpu.channels() != 3) {
            throw std::runtime_error("Expected BGR frame for GPU register/detect.");
        }

        frame_count_ += 1;
        const auto t_reg_start = now();
        cv::cuda::GpuMat reg_gpu = register_frame_gpu(frame_cpu);
        if (timing_enabled_) {
            timing_["registration"] += elapsed_seconds(t_reg_start, now());
        }

        if (pending_bg_reset_) {
            reset_background_model();
            pending_bg_reset_ = false;
        }

        detect_from_registered(reg_gpu);

        // CPU contour extraction from the downloaded low-res mask.
        std::vector<cv::Rect> detections = finalize_detections();
        if (return_registered_ && registered_out != nullptr) {
            if (timing_enabled_) {
                const auto t_dl_start = now();
                reg_gpu.download(registered_cpu_);
                timing_["download"] += elapsed_seconds(t_dl_start, now());
            } else {
                reg_gpu.download(registered_cpu_);
            }
            *registered_out = registered_cpu_.clone();
        }
        if (timing_enabled_) {
            const auto end = now();
            timing_["total"] += elapsed_seconds(start, end);
            timing_["calls"] += 1.0;
        }

        return detections;
    }

    cv::Mat get_last_mask() {
        if (mask_cpu_.empty()) {
            return cv::Mat();
        }
        cv::Mat mask_out;
        if (detect_scale_ == 1.0f) {
            mask_out = mask_cpu_;
        } else {
            cv::resize(mask_cpu_, mask_out, cv::Size(cropped_w_, cropped_h_), 0.0, 0.0, cv::INTER_NEAREST);
        }
        return mask_out;
    }

    cv::Mat get_last_cc_mask() {
        if (cc_mask_cpu_.empty()) {
            return cv::Mat();
        }
        return cc_mask_cpu_;
    }

    cv::Mat get_last_bb_mask() {
        if (bb_mask_cpu_.empty()) {
            return cv::Mat();
        }
        return bb_mask_cpu_;
    }

    std::vector<int> get_reference_events() const {
        return reference_events_;
    }

    std::pair<float, float> get_last_shift() const {
        return {last_dx_, last_dy_};
    }

    TranslationRegistrationDebug get_last_registration_debug() const {
        TranslationRegistrationDebug out;
        out.phase_response_threshold = phase_response_threshold_;
        out.response = last_response_;
        out.phase_ok = last_phase_ok_;
        out.dx = last_dx_;
        out.dy = last_dy_;
        out.reanchor_reason = last_reanchor_reason_;
        return out;
    }

private:

    void set_geometry(const cv::Mat &reference) {
        h_ = reference.rows;
        w_ = reference.cols;
        margin_ = static_cast<int>(std::min(h_, w_) * 0.01f);
        cropped_h_ = h_ - margin_ * 2;
        cropped_w_ = w_ - margin_ * 2;
    }

    void set_reference_mat(const cv::Mat &reference, bool record_event) {
        gray_ref_gpu_ = prepare_gray(reference);
        ref_fft_gpu_.release();
        if (phase_use_cached_fft_) {
            cv::cuda::dft(gray_ref_gpu_, ref_fft_gpu_, gray_ref_gpu_.size());
        }
        if (record_event) {
            reference_events_.push_back(static_cast<int>(frame_count_ - 1));
            if (reanchor_boost_frames_ > 0) {
                reanchor_boost_left_ = reanchor_boost_frames_;
            }
            log_reanchor_event();
        }
    }

    void set_reference_gray(const cv::cuda::GpuMat &gray_f) {
        gray_ref_gpu_ = gray_f.clone();
        ref_fft_gpu_.release();
        if (phase_use_cached_fft_) {
            cv::cuda::dft(gray_ref_gpu_, ref_fft_gpu_, gray_ref_gpu_.size());
        }
    }

    cv::cuda::GpuMat prepare_gray(const cv::Mat &frame_cpu) {
        // Upload and convert to float32 grayscale (optionally downscaled) for phase correlation.
        if (timing_enabled_) {
            const auto t_up_start = now();
            frame_gpu_.upload(frame_cpu);
            timing_["upload"] += elapsed_seconds(t_up_start, now());
        } else {
            frame_gpu_.upload(frame_cpu);
        }
        if (input_is_gray_) {
            gray_gpu_ = frame_gpu_;
        } else {
            cv::cuda::cvtColor(frame_gpu_, gray_gpu_, cv::COLOR_BGR2GRAY);
        }
        cv::cuda::GpuMat scaled;
        if (downscale_factor_ != 1.0f) {
            cv::cuda::resize(
                gray_gpu_,
                scaled,
                cv::Size(),
                downscale_factor_,
                downscale_factor_,
                cv::INTER_AREA
            );
        } else {
            scaled = gray_gpu_;
        }
        cv::cuda::GpuMat gray_f;
        scaled.convertTo(gray_f, CV_32F);
        return gray_f;
    }

    bool phase_correlate_gpu(
        const cv::cuda::GpuMat &gray_gpu,
        float &dx,
        float &dy,
        float &response
    ) {
        // GPU phase correlation: compute cross-power spectrum and locate peak shift.
        if (gray_ref_gpu_.empty()) {
            return false;
        }
        cv::cuda::GpuMat ref_fft;
        if (phase_use_cached_fft_) {
            if (ref_fft_gpu_.empty()) {
                cv::cuda::dft(gray_ref_gpu_, ref_fft_gpu_, gray_ref_gpu_.size());
            }
            ref_fft = ref_fft_gpu_;
        } else {
            cv::cuda::dft(gray_ref_gpu_, ref_fft, gray_ref_gpu_.size());
        }

        cv::cuda::GpuMat cur_fft;
        cv::cuda::dft(gray_gpu, cur_fft, gray_gpu.size());

        cv::cuda::GpuMat cps;
        cv::cuda::mulSpectrums(cur_fft, ref_fft, cps, 0, true);

        std::vector<cv::cuda::GpuMat> planes;
        cv::cuda::split(cps, planes);

        cv::cuda::GpuMat mag;
        cv::cuda::magnitude(planes[0], planes[1], mag);
        cv::cuda::addWeighted(mag, 1.0, mag, 0.0, 1e-9, mag);
        cv::cuda::divide(planes[0], mag, planes[0]);
        cv::cuda::divide(planes[1], mag, planes[1]);
        cv::cuda::merge(planes, cps);

        cv::cuda::GpuMat corr;
        cv::cuda::dft(
            cps,
            corr,
            cps.size(),
            cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT
        );

        cv::Mat corr_cpu;
        if (timing_enabled_) {
            const auto t_dl_start = now();
            corr.download(corr_cpu);
            timing_["download"] += elapsed_seconds(t_dl_start, now());
        } else {
            corr.download(corr_cpu);
        }
        double max_val = 0.0;
        cv::Point max_loc;
        cv::minMaxLoc(corr_cpu, nullptr, &max_val, nullptr, &max_loc);
        const int h = corr_cpu.rows;
        const int w = corr_cpu.cols;
        const float shift_x = (max_loc.x <= w / 2) ? static_cast<float>(max_loc.x)
                                                   : static_cast<float>(max_loc.x - w);
        const float shift_y = (max_loc.y <= h / 2) ? static_cast<float>(max_loc.y)
                                                   : static_cast<float>(max_loc.y - h);
        dx = shift_x / downscale_factor_;
        dy = shift_y / downscale_factor_;
        response = static_cast<float>(max_val);
        last_response_ = response;
        return true;
    }

    cv::cuda::GpuMat register_frame_gpu(const cv::Mat &frame_cpu) {
        // Compute translation and warp on GPU.
        float dx = 0.0f;
        float dy = 0.0f;
        float response = 0.0f;
        cv::cuda::GpuMat gray_f = prepare_gray(frame_cpu);
        last_phase_ok_ = false;
        last_reanchor_reason_ = "none";
        bool phase_ok = phase_correlate_gpu(gray_f, dx, dy, response) &&
                        response >= phase_response_threshold_;
        if (phase_ok && max_shift_ > 0.0f) {
            if (std::abs(dx) > max_shift_ || std::abs(dy) > max_shift_) {
                phase_ok = false;
                last_reanchor_reason_ = "max_shift";
            }
        }
        bool ok = phase_ok;
        last_phase_ok_ = phase_ok;
        if (!ok) {
            if (!phase_ok) {
                last_reanchor_reason_ = "phase";
            }
        }
        if (!ok) {
            last_dx_ = 0.0f;
            last_dy_ = 0.0f;
            if (last_reanchor_reason_ == "none") {
                last_reanchor_reason_ = "fallback_fail";
            }
            reg_gpu_ = crop_gpu(frame_gpu_);
        } else {
            last_dx_ = dx;
            last_dy_ = dy;
            cv::Mat warp = (cv::Mat_<float>(2, 3) << 1.0f, 0.0f, dx, 0.0f, 1.0f, dy);
            cv::cuda::warpAffine(
                frame_gpu_,
                reg_gpu_,
                warp,
                cv::Size(w_, h_),
                cv::INTER_LINEAR
            );
            reg_gpu_ = crop_gpu(reg_gpu_);
        }
        if (should_update_reference()) {
            set_reference_gray(gray_f);
            record_reference_event(ok ? "window" : "window_fallback");
            pending_bg_reset_ = true;
        }
        return reg_gpu_;
    }

    cv::cuda::GpuMat crop_gpu(const cv::cuda::GpuMat &mat) const {
        if (margin_ == 0) {
            return mat;
        }
        cv::Rect roi(margin_, margin_, w_ - margin_ * 2, h_ - margin_ * 2);
        return mat(roi);
    }

    bool touches_border(const cv::Rect &bbox) const {
        if (bbox.width <= 0 || bbox.height <= 0) {
            return true;
        }
        if (border_gate_percent_ <= 0.0f) {
            return false;
        }
        int margin_x = static_cast<int>(std::round(cropped_w_ * (border_gate_percent_ / 100.0f)));
        int margin_y = static_cast<int>(std::round(cropped_h_ * (border_gate_percent_ / 100.0f)));
        return (
            bbox.x <= margin_x ||
            bbox.y <= margin_y ||
            (bbox.x + bbox.width) >= (cropped_w_ - margin_x) ||
            (bbox.y + bbox.height) >= (cropped_h_ - margin_y)
        );
    }

    void log_reanchor_event() {
        if (!reanchor_log_.is_open()) {
            return;
        }
        reanchor_log_
            << frame_count_ << ","
            << last_response_ << ","
            << (last_phase_ok_ ? 1 : 0) << ","
            << last_dx_ << ","
            << last_dy_ << ","
            << last_reanchor_reason_
            << "\n";
        reanchor_log_.flush();
    }

    bool should_update_reference() const {
        return reference_window_frames_ > 0 && (frame_count_ % reference_window_frames_ == 0);
    }

    void record_reference_event(const std::string &reason) {
        last_reanchor_reason_ = reason;
        reference_events_.push_back(static_cast<int>(frame_count_ - 1));
        if (reanchor_boost_frames_ > 0) {
            reanchor_boost_left_ = reanchor_boost_frames_;
        }
        pending_bg_reset_ = true;
        log_reanchor_event();
    }

    void reset_background_model() {
        backsub_ = cv::cuda::createBackgroundSubtractorMOG2(
            mog2_history_,
            mog2_var_threshold_,
            false
        );
        mask_gpu_.release();
        mask_cpu_.release();
        cc_mask_cpu_.release();
        bb_mask_cpu_.release();
    }

    void init_detector_buffers() {
        int type = input_is_gray_ ? CV_8UC1 : CV_8UC3;
        if (detect_scale_ != 1.0f) {
            scaled_h_ = std::max(1, static_cast<int>(std::round(cropped_h_ * detect_scale_)));
            scaled_w_ = std::max(1, static_cast<int>(std::round(cropped_w_ * detect_scale_)));
        } else {
            scaled_h_ = cropped_h_;
            scaled_w_ = cropped_w_;
        }
        frame_small_gpu_.create(scaled_h_, scaled_w_, type);
        blur_small_gpu_.create(scaled_h_, scaled_w_, type);
        fg_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        mask_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        mask_cpu_.create(scaled_h_, scaled_w_, CV_8UC1);
    }

    void detect_from_registered(const cv::cuda::GpuMat &reg_gpu) {
        // Detection pipeline on the registered GPU frame.
        const auto start = now();
        upload_and_scale(reg_gpu);
        gaussian_blur();
        const auto t_blur = now();
        if (timing_enabled_) {
            timing_["blur"] += elapsed_seconds(start, t_blur);
        }

        float lr = learning_rate_;
        if (reanchor_boost_left_ > 0) {
            lr = reanchor_boost_lr_;
        }
        backsub_->apply(blur_small_gpu_, fg_gpu_, lr);
        const auto t_fg = now();
        if (timing_enabled_) {
            timing_["bgsub"] += elapsed_seconds(t_blur, t_fg);
        }

        fg_gpu_.copyTo(mask_gpu_);
        if (reanchor_boost_left_ > 0) {
            reanchor_boost_left_ -= 1;
        }

        // contours timing is tracked in finalize_detections (CPU)
    }

    void upload_and_scale(const cv::cuda::GpuMat &frame_gpu) {
        // Downscale registered frame for detector if requested.
        if (detect_scale_ == 1.0f) {
            frame_small_gpu_ = frame_gpu;
            return;
        }
        cv::cuda::resize(
            frame_gpu,
            frame_small_gpu_,
            cv::Size(scaled_w_, scaled_h_),
            0.0,
            0.0,
            cv::INTER_AREA
        );
    }

    void gaussian_blur() {
        // Apply prebuilt Gaussian filter on the scaled frame.
        gaussian_filter_->apply(frame_small_gpu_, blur_small_gpu_);
    }


    std::vector<cv::Rect> finalize_detections() {
        // CPU contour extraction from the downloaded low-res mask.
        if (timing_enabled_) {
            const auto t_dl_start = now();
            mask_gpu_.download(mask_cpu_);
            timing_["download"] += elapsed_seconds(t_dl_start, now());
        } else {
            mask_gpu_.download(mask_cpu_);
        }

        if (close_iterations_ > 0 || open_iterations_ > 0) {
            if (morph_kernel_.empty()) {
                morph_kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            }
            if (close_iterations_ > 0) {
                cv::morphologyEx(
                    mask_cpu_,
                    mask_cpu_,
                    cv::MORPH_CLOSE,
                    morph_kernel_,
                    cv::Point(-1, -1),
                    close_iterations_
                );
            }
            if (open_iterations_ > 0) {
                cv::morphologyEx(
                    mask_cpu_,
                    mask_cpu_,
                    cv::MORPH_OPEN,
                    morph_kernel_,
                    cv::Point(-1, -1),
                    open_iterations_
                );
            }
        }
        if (dilate_iterations_ > 0) {
            if (morph_kernel_.empty()) {
                morph_kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            }
            cv::dilate(mask_cpu_, mask_cpu_, morph_kernel_, cv::Point(-1, -1), dilate_iterations_);
        }

        std::vector<std::vector<cv::Point>> contours;
        const auto t_contours_start = now();
        cv::findContours(mask_cpu_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (timing_enabled_) {
            const auto t_contours_end = now();
            timing_["contours"] += elapsed_seconds(t_contours_start, t_contours_end);
        }

        cc_mask_cpu_ = cv::Mat::zeros(mask_cpu_.size(), CV_8UC3);
        if (!contours.empty()) {
            for (size_t i = 0; i < contours.size(); ++i) {
                cv::Scalar color(
                    static_cast<double>((37 * i) % 255),
                    static_cast<double>((17 * i) % 255),
                    static_cast<double>((97 * i) % 255)
                );
                cv::drawContours(cc_mask_cpu_, contours, static_cast<int>(i), color, cv::FILLED);
            }
        }
        if (detect_scale_ != 1.0f) {
            cv::resize(cc_mask_cpu_, cc_mask_cpu_, cv::Size(cropped_w_, cropped_h_), 0.0, 0.0, cv::INTER_NEAREST);
        }

        const auto t_cpu_start = now();
        std::vector<cv::Rect> detections;
        detections.reserve(contours.size());
        for (const auto &c : contours) {
            double area = cv::contourArea(c);
            if (!(min_contour_area_ < area && area < max_contour_area_)) {
                continue;
            }
            cv::Rect bbox = cv::boundingRect(c);
            if (bbox.width <= 0 || bbox.height <= 0) {
                continue;
            }
            double aspect = static_cast<double>(bbox.width) / static_cast<double>(bbox.height);
            if (!(0.2 <= aspect && aspect <= 5.0)) {
                continue;
            }
            if (detect_scale_ != 1.0f) {
                int x = static_cast<int>(std::round(bbox.x * scale_inv_));
                int y = static_cast<int>(std::round(bbox.y * scale_inv_));
                int w = static_cast<int>(std::round(bbox.width * scale_inv_));
                int h = static_cast<int>(std::round(bbox.height * scale_inv_));
                bbox = cv::Rect(x, y, w, h);
            }
            if (touches_border(bbox)) {
                continue;
            }
            detections.push_back(bbox);
        }

        if (cc_mask_cpu_.channels() == 3) {
            bb_mask_cpu_ = cc_mask_cpu_.clone();
        } else {
            bb_mask_cpu_ = cv::Mat::zeros(cv::Size(cropped_w_, cropped_h_), CV_8UC3);
        }
        for (const auto &bbox : detections) {
            cv::rectangle(bb_mask_cpu_, bbox, cv::Scalar(0, 0, 255), 1);
        }

        if (timing_enabled_) {
            const auto t_cpu_end = now();
            timing_["bbox_creation"] += elapsed_seconds(t_cpu_start, t_cpu_end);
        }
        return detections;
    }

    static std::chrono::steady_clock::time_point now() {
        return std::chrono::steady_clock::now();
    }

    static double elapsed_seconds(const std::chrono::steady_clock::time_point &start,
                                  const std::chrono::steady_clock::time_point &end) {
        return std::chrono::duration<double>(end - start).count();
    }

    bool input_is_gray_ = true;
    float downscale_factor_ = 1.0f;
    float detect_scale_ = 1.0f;
    float scale_inv_ = 1.0f;
    float learning_rate_ = 0.05f;
    int reference_window_frames_ = 1;
    int mog2_history_ = 400;
    float mog2_var_threshold_ = 20.0f;
    float phase_response_threshold_ = 0.1f;
    int reanchor_boost_frames_ = 0;
    int reanchor_boost_left_ = 0;
    float max_shift_ = 0.0f;
    float reanchor_boost_lr_ = 0.0f;
    bool phase_use_cached_fft_ = true;
    std::ofstream reanchor_log_;
    float last_response_ = 0.0f;
    bool last_phase_ok_ = false;
    std::string last_reanchor_reason_ = "none";
    bool pending_bg_reset_ = false;

    bool timing_enabled_ = true;
    std::map<std::string, double> timing_;

    int h_ = 0;
    int w_ = 0;
    int margin_ = 0;
    int cropped_h_ = 0;
    int cropped_w_ = 0;
    int scaled_h_ = 0;
    int scaled_w_ = 0;

    std::size_t frame_count_ = 0;

    cv::cuda::GpuMat frame_gpu_;
    cv::cuda::GpuMat gray_gpu_;
    cv::cuda::GpuMat gray_ref_gpu_;
    cv::cuda::GpuMat ref_fft_gpu_;
    cv::cuda::GpuMat reg_gpu_;

    cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> backsub_;
    cv::Ptr<cv::cuda::Filter> gaussian_filter_;
    cv::Mat morph_kernel_;
    int close_iterations_ = 1;
    int open_iterations_ = 1;
    int dilate_iterations_ = 2;
    int min_contour_area_ = 50;
    int max_contour_area_ = 1500;
    float border_gate_percent_ = 1.0f;

    cv::cuda::GpuMat frame_small_gpu_;
    cv::cuda::GpuMat blur_small_gpu_;
    cv::cuda::GpuMat fg_gpu_;
    cv::cuda::GpuMat mask_gpu_;

    cv::Mat mask_cpu_;
    cv::Mat cc_mask_cpu_;
    cv::Mat bb_mask_cpu_;
    cv::Mat registered_cpu_;
    std::vector<int> reference_events_;
    float last_dx_ = 0.0f;
    float last_dy_ = 0.0f;
    bool return_registered_ = true;

};

class RegisterAndDetectHomographyGPU {
public:
    RegisterAndDetectHomographyGPU(
        const cv::Mat &reference,
        bool input_is_gray,
        float downscale_factor = 1.0f,
        float detect_scale = 1.0f,
        float learning_rate = 0.05f,
        int reference_window_frames = 1,
        float mog2_var_threshold = 20.0f,
        int min_contour_area = 50,
        int max_contour_area = 1500,
        float border_gate_percent = 1.0f,
        const std::string &feature_type = "ORB",
        const std::string &matcher_type = "BF",
        float knn_ratio = 0.75f,
        float ransac_reproj_threshold = 5.0f,
        int min_inliers = 0,
        int reanchor_boost_frames = 0,
        float reanchor_boost_lr = 0.0f,
        bool enable_timing = true,
        bool return_registered = true
    )
        : input_is_gray_(input_is_gray),
          downscale_factor_(downscale_factor > 0.0f ? downscale_factor : 1.0f),
          detect_scale_(detect_scale > 0.0f ? detect_scale : 1.0f),
          scale_inv_(1.0f / detect_scale_),
          learning_rate_(learning_rate),
          reference_window_frames_(reference_window_frames > 0 ? reference_window_frames : 1),
          mog2_history_(reference_window_frames_),
          mog2_var_threshold_(mog2_var_threshold),
          min_contour_area_(std::max(0, min_contour_area)),
          max_contour_area_(std::max(min_contour_area_, max_contour_area)),
          border_gate_percent_(std::clamp(border_gate_percent, 0.0f, 49.0f)),
          feature_type_(feature_type),
          matcher_type_(matcher_type),
          knn_ratio_(knn_ratio),
          ransac_reproj_threshold_(ransac_reproj_threshold),
          min_inliers_(min_inliers > 0 ? min_inliers : 0),
          reanchor_boost_frames_(reanchor_boost_frames > 0 ? reanchor_boost_frames : 0),
          reanchor_boost_lr_(reanchor_boost_lr > 0.0f ? reanchor_boost_lr : learning_rate_),
          timing_enabled_(enable_timing),
          return_registered_(return_registered) {
        if (reference.empty()) {
            throw std::runtime_error("Expected non-empty reference frame.");
        }
        cv::Mat ref_mat = reference;
        set_geometry(ref_mat);
        set_reference_mat(ref_mat, false, "init");
        init_detector_buffers();

        backsub_ = cv::cuda::createBackgroundSubtractorMOG2(
            mog2_history_,
            mog2_var_threshold_,
            false
        );

        int type = input_is_gray_ ? CV_8UC1 : CV_8UC3;
        gaussian_filter_ = cv::cuda::createGaussianFilter(
            type,
            type,
            cv::Size(3, 3),
            0.0
        );

        if (timing_enabled_) {
            timing_ = {
                {"calls", 0.0},
                {"registration", 0.0},
                {"keypoints", 0.0},
                {"match", 0.0},
                {"homography", 0.0},
                {"warp", 0.0},
                {"blur", 0.0},
                {"bgsub", 0.0},
                {"contours", 0.0},
                {"upload", 0.0},
                {"download", 0.0},
                {"bbox_creation", 0.0},
                {"total", 0.0},
            };
        }
    }

    static bool is_available() {
        try {
            return cv::cuda::getCudaEnabledDeviceCount() > 0;
        } catch (const cv::Exception &) {
            return false;
        }
    }

    TimingSummary timing_summary(bool reset = false) {
        TimingSummary out;
        if (!timing_enabled_) {
            out.enabled = false;
            return out;
        }
        const double calls = timing_["calls"];
        out.enabled = true;
        out.calls = static_cast<int>(calls);
        if (calls > 0.0) {
            out.values["total_ms_avg"] = (timing_["total"] / calls) * 1000.0;
            out.values["registration_ms_avg"] = (timing_["registration"] / calls) * 1000.0;
            out.values["keypoints_ms_avg"] = (timing_["keypoints"] / calls) * 1000.0;
            out.values["match_ms_avg"] = (timing_["match"] / calls) * 1000.0;
            out.values["homography_ms_avg"] = (timing_["homography"] / calls) * 1000.0;
            out.values["warp_ms_avg"] = (timing_["warp"] / calls) * 1000.0;
            out.values["blur_ms_avg"] = (timing_["blur"] / calls) * 1000.0;
            out.values["bgsub_ms_avg"] = (timing_["bgsub"] / calls) * 1000.0;
            out.values["contours_ms_avg"] = (timing_["contours"] / calls) * 1000.0;
            out.values["upload_ms_avg"] = (timing_["upload"] / calls) * 1000.0;
            out.values["download_ms_avg"] = (timing_["download"] / calls) * 1000.0;
            out.values["bbox_creation_ms_avg"] = (timing_["bbox_creation"] / calls) * 1000.0;
            out.values["re_registration_calls"] = static_cast<int>(re_registration_calls_);
        }
        if (reset) {
            reset_timing();
        }
        return out;
    }

    void reset_timing() {
        if (!timing_enabled_) {
            return;
        }
        for (auto &kv : timing_) {
            kv.second = 0.0;
        }
    }

    std::vector<cv::Rect> register_and_detect(const cv::Mat &frame, cv::Mat *registered_out = nullptr) {
        const auto start = now();
        if (frame.empty()) {
            throw std::runtime_error("Expected non-empty frame for GPU register/detect.");
        }
        cv::Mat frame_cpu = frame;
        if (input_is_gray_ && frame_cpu.channels() != 1) {
            throw std::runtime_error("Expected grayscale frame for GPU register/detect.");
        }
        if (!input_is_gray_ && frame_cpu.channels() != 3) {
            throw std::runtime_error("Expected BGR frame for GPU register/detect.");
        }

        frame_count_ += 1;
        const auto t_reg_start = now();
        cv::cuda::GpuMat reg_gpu = register_frame_gpu(frame_cpu);
        if (timing_enabled_) {
            timing_["registration"] += elapsed_seconds(t_reg_start, now());
        }

        if (pending_bg_reset_) {
            reset_background_model();
            pending_bg_reset_ = false;
        }

        detect_from_registered(reg_gpu);

        std::vector<cv::Rect> detections = finalize_detections();
        if (return_registered_ && registered_out != nullptr) {
            if (timing_enabled_) {
                const auto t_dl_start = now();
                reg_gpu.download(registered_cpu_);
                timing_["download"] += elapsed_seconds(t_dl_start, now());
            } else {
                reg_gpu.download(registered_cpu_);
            }
            *registered_out = registered_cpu_.clone();
        }

        if (timing_enabled_) {
            const auto end = now();
            timing_["total"] += elapsed_seconds(start, end);
            timing_["calls"] += 1.0;
        }

        return detections;
    }

    cv::Mat get_last_mask() {
        if (mask_cpu_.empty()) {
            return cv::Mat();
        }
        cv::Mat mask_out;
        if (detect_scale_ == 1.0f) {
            mask_out = mask_cpu_;
        } else {
            cv::resize(mask_cpu_, mask_out, cv::Size(cropped_w_, cropped_h_), 0.0, 0.0, cv::INTER_NEAREST);
        }
        return mask_out;
    }

    cv::Mat get_last_cc_mask() {
        if (cc_mask_cpu_.empty()) {
            return cv::Mat();
        }
        return cc_mask_cpu_;
    }

    cv::Mat get_last_bb_mask() {
        if (bb_mask_cpu_.empty()) {
            return cv::Mat();
        }
        return bb_mask_cpu_;
    }

    std::vector<int> get_reference_events() const {
        return reference_events_;
    }

    std::pair<float, float> get_last_shift() const {
        return {0.0f, 0.0f};
    }

    HomographyRegistrationDebug get_last_registration_debug() const {
        HomographyRegistrationDebug out;
        out.fail_reason = last_fail_reason_;
        out.match_count = last_match_count_;
        out.inlier_count = last_inlier_count_;
        out.overlap_ratio = last_overlap_;
        out.reference_reason = last_reference_reason_;
        return out;
    }

private:

    void set_geometry(const cv::Mat &reference) {
        h_ = reference.rows;
        w_ = reference.cols;
        margin_ = static_cast<int>(std::min(h_, w_) * 0.01f);
        cropped_h_ = h_ - margin_ * 2;
        cropped_w_ = w_ - margin_ * 2;

        float s = downscale_factor_;
        scale_matrix_ = (cv::Mat_<float>(3, 3) << s, 0.0f, 0.0f, 0.0f, s, 0.0f, 0.0f, 0.0f, 1.0f);
        scale_matrix_inv_ = (cv::Mat_<float>(3, 3) << 1.0f / s, 0.0f, 0.0f, 0.0f, 1.0f / s, 0.0f, 0.0f, 0.0f, 1.0f);

        frame_corners_ = {
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(static_cast<float>(w_), 0.0f),
            cv::Point2f(static_cast<float>(w_), static_cast<float>(h_)),
            cv::Point2f(0.0f, static_cast<float>(h_))
        };
        frame_area_ = static_cast<float>(w_ * h_);
    }

    void set_reference_mat(const cv::Mat &reference, bool record_event, const std::string &reason) {
        cv::Mat gray = prepare_gray(reference);
        build_feature_extractor();
        extract_features(gray, kp_ref_, descriptors_ref_);
        gray_ref_ = gray;
        if (record_event) {
            reference_events_.push_back(static_cast<int>(frame_count_ - 1));
            last_reference_reason_ = reason;
            if (reanchor_boost_frames_ > 0) {
                reanchor_boost_left_ = reanchor_boost_frames_;
            }
            pending_bg_reset_ = true;
        }
    }

    void set_reference_from_features(
        const cv::Mat &gray,
        const std::vector<cv::KeyPoint> &kp,
        const cv::Mat &descriptors,
        const std::string &reason
    ) {
        gray_ref_ = gray;
        kp_ref_ = kp;
        descriptors_ref_ = descriptors.clone();
        reference_events_.push_back(static_cast<int>(frame_count_ - 1));
        last_reference_reason_ = reason;
        if (reanchor_boost_frames_ > 0) {
            reanchor_boost_left_ = reanchor_boost_frames_;
        }
        pending_bg_reset_ = true;
    }

    cv::Mat prepare_gray(const cv::Mat &frame) const {
        cv::Mat gray;
        if (input_is_gray_) {
            gray = frame;
        } else {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        }
        if (downscale_factor_ != 1.0f) {
            cv::Mat scaled;
            cv::resize(gray, scaled, cv::Size(), downscale_factor_, downscale_factor_, cv::INTER_AREA);
            return scaled;
        }
        return gray;
    }

    void build_feature_extractor() {
        std::string type = feature_type_;
        if (type.empty()) {
            type = "ORB";
        }
        std::string upper;
        upper.reserve(type.size());
        for (char c : type) {
            upper.push_back(static_cast<char>(std::toupper(c)));
        }
        feature_type_ = upper;

        if (feature_type_ == "SIFT") {
            feature_extractor_ = cv::SIFT::create(500);
            descriptor_extractor_.release();
            norm_type_ = cv::NORM_L2;
        } else if (feature_type_ == "AKAZE") {
            feature_extractor_ = cv::AKAZE::create();
            descriptor_extractor_.release();
            norm_type_ = cv::NORM_HAMMING;
        } else if (feature_type_ == "BRISK") {
            feature_extractor_ = cv::BRISK::create();
            descriptor_extractor_.release();
            norm_type_ = cv::NORM_HAMMING;
        } else if (feature_type_ == "FAST_BRIEF") {
            fast_detector_ = cv::FastFeatureDetector::create(20, true);
            descriptor_extractor_ = cv::xfeatures2d::BriefDescriptorExtractor::create(32);
            feature_extractor_.release();
            norm_type_ = cv::NORM_HAMMING;
        } else {
            feature_extractor_ = cv::ORB::create(500);
            descriptor_extractor_.release();
            norm_type_ = cv::NORM_HAMMING;
        }

        std::string matcher_type = matcher_type_;
        if (matcher_type.empty()) {
            matcher_type = "BF";
        }
        std::string matcher_upper;
        matcher_upper.reserve(matcher_type.size());
        for (char c : matcher_type) {
            matcher_upper.push_back(static_cast<char>(std::toupper(c)));
        }
        matcher_type_ = matcher_upper;
    }

    void extract_features(const cv::Mat &gray,
                          std::vector<cv::KeyPoint> &kps,
                          cv::Mat &descriptors) {
        if (feature_type_ == "FAST_BRIEF") {
            if (!fast_detector_ || !descriptor_extractor_) {
                throw std::runtime_error("FAST_BRIEF requires xfeatures2d.");
            }
            fast_detector_->detect(gray, kps);
            if (kps.size() > 500) {
                std::nth_element(kps.begin(), kps.begin() + 500, kps.end(),
                                 [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
                                     return a.response > b.response;
                                 });
                kps.resize(500);
            }
            descriptor_extractor_->compute(gray, kps, descriptors);
        } else if (feature_extractor_) {
            feature_extractor_->detectAndCompute(gray, cv::noArray(), kps, descriptors);
        }
    }

    std::vector<cv::DMatch> match_descriptors(const cv::Mat &descriptors_frame) {
        std::vector<cv::DMatch> matches;
        if (descriptors_ref_.empty() || descriptors_frame.empty()) {
            return matches;
        }
        if (matcher_type_ == "KNN") {
            cv::BFMatcher matcher(norm_type_);
            std::vector<std::vector<cv::DMatch>> knn;
            matcher.knnMatch(descriptors_frame, descriptors_ref_, knn, 2);
            for (const auto &pair : knn) {
                if (pair.size() < 2) {
                    continue;
                }
                if (pair[0].distance < knn_ratio_ * pair[1].distance) {
                    matches.push_back(pair[0]);
                }
            }
        } else if (matcher_type_ == "FLANN") {
            cv::FlannBasedMatcher matcher;
            std::vector<std::vector<cv::DMatch>> knn;
            matcher.knnMatch(descriptors_frame, descriptors_ref_, knn, 2);
            for (const auto &pair : knn) {
                if (pair.size() < 2) {
                    continue;
                }
                if (pair[0].distance < knn_ratio_ * pair[1].distance) {
                    matches.push_back(pair[0]);
                }
            }
        } else {
            cv::BFMatcher matcher(norm_type_, true);
            matcher.match(descriptors_frame, descriptors_ref_, matches);
            std::sort(matches.begin(), matches.end(),
                      [](const cv::DMatch &a, const cv::DMatch &b) {
                          return a.distance < b.distance;
                      });
        }
        return matches;
    }

    cv::Mat calc_homography(const std::vector<cv::KeyPoint> &kp_frame,
                            const std::vector<cv::DMatch> &matches) {
        if (matches.size() < 4) {
            last_inlier_count_ = 0;
            return cv::Mat();
        }
        std::vector<cv::Point2f> pts1;
        std::vector<cv::Point2f> pts2;
        pts1.reserve(matches.size());
        pts2.reserve(matches.size());
        for (const auto &m : matches) {
            pts1.push_back(kp_frame[m.queryIdx].pt);
            pts2.push_back(kp_ref_[m.trainIdx].pt);
        }
        cv::Mat mask;
        cv::Mat H = cv::findHomography(
            pts1,
            pts2,
            cv::RANSAC,
            ransac_reproj_threshold_,
            mask
        );
        if (!mask.empty()) {
            last_inlier_count_ = static_cast<int>(cv::countNonZero(mask));
        } else {
            last_inlier_count_ = 0;
        }
        return H;
    }

    float overlap_ratio(const cv::Mat &H) const {
        std::vector<cv::Point2f> dst;
        cv::perspectiveTransform(frame_corners_, dst, H);
        std::vector<cv::Point2f> hull;
        cv::convexHull(dst, hull);
        std::vector<cv::Point2f> ref_hull;
        cv::convexHull(frame_corners_, ref_hull);
        double inter_area = cv::intersectConvexConvex(hull, ref_hull, hull);
        if (inter_area <= 0.0) {
            return 0.0f;
        }
        return static_cast<float>(inter_area / frame_area_);
    }

    cv::Mat scale_homography(const cv::Mat &H) const {
        if (downscale_factor_ == 1.0f) {
            return H;
        }
        cv::Mat H32;
        if (H.type() != CV_32F) {
            H.convertTo(H32, CV_32F);
        } else {
            H32 = H;
        }
        return scale_matrix_inv_ * H32 * scale_matrix_;
    }

    cv::cuda::GpuMat register_frame_gpu(const cv::Mat &frame_mat) {
        auto start = now();
        cv::Mat gray = prepare_gray(frame_mat);
        auto t_gray = now();
        std::vector<cv::KeyPoint> kp_frame;
        cv::Mat descriptors_frame;
        auto keypoint_start = t_gray;
        extract_features(gray, kp_frame, descriptors_frame);
        if (timing_enabled_) {
            timing_["gray"] += elapsed_seconds(start, t_gray);
            timing_["keypoints"] += elapsed_seconds(keypoint_start, now());
        }

        if (timing_enabled_) {
            const auto t_up_start = now();
            frame_gpu_.upload(frame_mat);
            timing_["upload"] += elapsed_seconds(t_up_start, now());
        } else {
            frame_gpu_.upload(frame_mat);
        }

        cv::cuda::GpuMat reg_gpu;
        bool ok = false;
        last_fail_reason_.clear();
        last_match_count_ = 0;
        last_overlap_ = 0.0f;

        if (descriptors_frame.empty() || kp_frame.size() < 4) {
            last_fail_reason_ = "descriptors";
        } else {
            auto match_start = now();
            std::vector<cv::DMatch> matches = match_descriptors(descriptors_frame);
            last_match_count_ = static_cast<int>(matches.size());
            if (timing_enabled_) {
                timing_["match"] += elapsed_seconds(match_start, now());
            }
            if (matches.empty()) {
                last_fail_reason_ = "match";
            } else {
                auto h_start = now();
                cv::Mat H = calc_homography(kp_frame, matches);
                if (!H.empty()) {
                    H = scale_homography(H);
                }
                if (timing_enabled_) {
                    timing_["homography"] += elapsed_seconds(h_start, now());
                }
                if (H.empty()) {
                    last_fail_reason_ = "homography";
                } else if (min_inliers_ && last_inlier_count_ < min_inliers_) {
                    last_fail_reason_ = "inliers";
                } else {
                    last_overlap_ = overlap_ratio(H);
                    auto warp_start = now();
                    cv::cuda::warpPerspective(frame_gpu_, reg_gpu_, H, cv::Size(w_, h_), cv::INTER_LINEAR);
                    if (timing_enabled_) {
                        timing_["warp"] += elapsed_seconds(warp_start, now());
                    }
                    reg_gpu_ = crop_gpu(reg_gpu_);
                    ok = true;
                }
            }
        }

        if (!ok) {
            reg_gpu_ = crop_gpu(frame_gpu_);
        }

        if (descriptors_frame.rows >= 4 && should_update_reference()) {
            set_reference_from_features(gray, kp_frame, descriptors_frame, ok ? "window" : "window_fallback");
            pending_bg_reset_ = true;
            if (!ok) {
                re_registration_calls_ += 1;
            }
        }

        return reg_gpu_;
    }

    cv::cuda::GpuMat crop_gpu(const cv::cuda::GpuMat &mat) const {
        if (margin_ == 0) {
            return mat;
        }
        cv::Rect roi(margin_, margin_, w_ - margin_ * 2, h_ - margin_ * 2);
        return mat(roi);
    }

    bool touches_border(const cv::Rect &bbox) const {
        if (bbox.width <= 0 || bbox.height <= 0) {
            return true;
        }
        if (border_gate_percent_ <= 0.0f) {
            return false;
        }
        int margin_x = static_cast<int>(std::round(cropped_w_ * (border_gate_percent_ / 100.0f)));
        int margin_y = static_cast<int>(std::round(cropped_h_ * (border_gate_percent_ / 100.0f)));
        return (
            bbox.x <= margin_x ||
            bbox.y <= margin_y ||
            (bbox.x + bbox.width) >= (cropped_w_ - margin_x) ||
            (bbox.y + bbox.height) >= (cropped_h_ - margin_y)
        );
    }

    bool should_update_reference() const {
        return reference_window_frames_ > 0 && (frame_count_ % reference_window_frames_ == 0);
    }

    void reset_background_model() {
        backsub_ = cv::cuda::createBackgroundSubtractorMOG2(
            mog2_history_,
            mog2_var_threshold_,
            false
        );
        mask_gpu_.release();
        mask_cpu_.release();
        cc_mask_cpu_.release();
        bb_mask_cpu_.release();
    }

    void init_detector_buffers() {
        int type = input_is_gray_ ? CV_8UC1 : CV_8UC3;
        if (detect_scale_ != 1.0f) {
            scaled_h_ = std::max(1, static_cast<int>(std::round(cropped_h_ * detect_scale_)));
            scaled_w_ = std::max(1, static_cast<int>(std::round(cropped_w_ * detect_scale_)));
        } else {
            scaled_h_ = cropped_h_;
            scaled_w_ = cropped_w_;
        }
        frame_small_gpu_.create(scaled_h_, scaled_w_, type);
        blur_small_gpu_.create(scaled_h_, scaled_w_, type);
        fg_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        mask_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        mask_cpu_.create(scaled_h_, scaled_w_, CV_8UC1);
    }

    void detect_from_registered(const cv::cuda::GpuMat &reg_gpu) {
        const auto start = now();
        upload_and_scale(reg_gpu);
        gaussian_blur();
        const auto t_blur = now();
        if (timing_enabled_) {
            timing_["blur"] += elapsed_seconds(start, t_blur);
        }

        float lr = learning_rate_;
        if (reanchor_boost_left_ > 0) {
            lr = reanchor_boost_lr_;
        }
        backsub_->apply(blur_small_gpu_, fg_gpu_, lr);
        const auto t_fg = now();
        if (timing_enabled_) {
            timing_["bgsub"] += elapsed_seconds(t_blur, t_fg);
        }

        fg_gpu_.copyTo(mask_gpu_);
        if (reanchor_boost_left_ > 0) {
            reanchor_boost_left_ -= 1;
        }
    }

    void upload_and_scale(const cv::cuda::GpuMat &frame_gpu) {
        if (detect_scale_ == 1.0f) {
            frame_small_gpu_ = frame_gpu;
            return;
        }
        cv::cuda::resize(
            frame_gpu,
            frame_small_gpu_,
            cv::Size(scaled_w_, scaled_h_),
            0.0,
            0.0,
            cv::INTER_AREA
        );
    }

    void gaussian_blur() {
        gaussian_filter_->apply(frame_small_gpu_, blur_small_gpu_);
    }


    std::vector<cv::Rect> finalize_detections() {
        if (timing_enabled_) {
            const auto t_dl_start = now();
            mask_gpu_.download(mask_cpu_);
            timing_["download"] += elapsed_seconds(t_dl_start, now());
        } else {
            mask_gpu_.download(mask_cpu_);
        }

        if (close_iterations_ > 0 || open_iterations_ > 0) {
            if (morph_kernel_.empty()) {
                morph_kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            }
            if (close_iterations_ > 0) {
                cv::morphologyEx(
                    mask_cpu_,
                    mask_cpu_,
                    cv::MORPH_CLOSE,
                    morph_kernel_,
                    cv::Point(-1, -1),
                    close_iterations_
                );
            }
            if (open_iterations_ > 0) {
                cv::morphologyEx(
                    mask_cpu_,
                    mask_cpu_,
                    cv::MORPH_OPEN,
                    morph_kernel_,
                    cv::Point(-1, -1),
                    open_iterations_
                );
            }
        }
        if (dilate_iterations_ > 0) {
            if (morph_kernel_.empty()) {
                morph_kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            }
            cv::dilate(mask_cpu_, mask_cpu_, morph_kernel_, cv::Point(-1, -1), dilate_iterations_);
        }

        std::vector<std::vector<cv::Point>> contours;
        const auto t_contours_start = now();
        cv::findContours(mask_cpu_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (timing_enabled_) {
            const auto t_contours_end = now();
            timing_["contours"] += elapsed_seconds(t_contours_start, t_contours_end);
        }

        cc_mask_cpu_ = cv::Mat::zeros(mask_cpu_.size(), CV_8UC3);
        if (!contours.empty()) {
            for (size_t i = 0; i < contours.size(); ++i) {
                cv::Scalar color(
                    static_cast<double>((37 * i) % 255),
                    static_cast<double>((17 * i) % 255),
                    static_cast<double>((97 * i) % 255)
                );
                cv::drawContours(cc_mask_cpu_, contours, static_cast<int>(i), color, cv::FILLED);
            }
        }
        if (detect_scale_ != 1.0f) {
            cv::resize(cc_mask_cpu_, cc_mask_cpu_, cv::Size(cropped_w_, cropped_h_), 0.0, 0.0, cv::INTER_NEAREST);
        }

        const auto t_cpu_start = now();
        std::vector<cv::Rect> detections;
        detections.reserve(contours.size());
        for (const auto &c : contours) {
            double area = cv::contourArea(c);
            if (!(min_contour_area_ < area && area < max_contour_area_)) {
                continue;
            }
            cv::Rect bbox = cv::boundingRect(c);
            if (bbox.width <= 0 || bbox.height <= 0) {
                continue;
            }
            double aspect = static_cast<double>(bbox.width) / static_cast<double>(bbox.height);
            if (!(0.2 <= aspect && aspect <= 5.0)) {
                continue;
            }
            if (detect_scale_ != 1.0f) {
                int x = static_cast<int>(std::round(bbox.x * scale_inv_));
                int y = static_cast<int>(std::round(bbox.y * scale_inv_));
                int w = static_cast<int>(std::round(bbox.width * scale_inv_));
                int h = static_cast<int>(std::round(bbox.height * scale_inv_));
                bbox = cv::Rect(x, y, w, h);
            }
            if (touches_border(bbox)) {
                continue;
            }
            detections.push_back(bbox);
        }

        if (cc_mask_cpu_.channels() == 3) {
            bb_mask_cpu_ = cc_mask_cpu_.clone();
        } else {
            bb_mask_cpu_ = cv::Mat::zeros(cv::Size(cropped_w_, cropped_h_), CV_8UC3);
        }
        for (const auto &bbox : detections) {
            cv::rectangle(bb_mask_cpu_, bbox, cv::Scalar(0, 0, 255), 1);
        }

        if (timing_enabled_) {
            const auto t_cpu_end = now();
            timing_["bbox_creation"] += elapsed_seconds(t_cpu_start, t_cpu_end);
        }
        return detections;
    }

    static std::chrono::steady_clock::time_point now() {
        return std::chrono::steady_clock::now();
    }

    static double elapsed_seconds(const std::chrono::steady_clock::time_point &start,
                                  const std::chrono::steady_clock::time_point &end) {
        return std::chrono::duration<double>(end - start).count();
    }

    bool input_is_gray_ = true;
    float downscale_factor_ = 1.0f;
    float detect_scale_ = 1.0f;
    float scale_inv_ = 1.0f;
    float learning_rate_ = 0.05f;
    int reference_window_frames_ = 1;
    int mog2_history_ = 400;
    float mog2_var_threshold_ = 20.0f;
    std::string feature_type_;
    std::string matcher_type_;
    float knn_ratio_ = 0.75f;
    float ransac_reproj_threshold_ = 5.0f;
    int min_inliers_ = 0;
    int reanchor_boost_frames_ = 0;
    int reanchor_boost_left_ = 0;
    float reanchor_boost_lr_ = 0.0f;

    bool timing_enabled_ = true;
    std::map<std::string, double> timing_;

    int h_ = 0;
    int w_ = 0;
    int margin_ = 0;
    int cropped_h_ = 0;
    int cropped_w_ = 0;
    int scaled_h_ = 0;
    int scaled_w_ = 0;

    std::size_t frame_count_ = 0;

    cv::Mat gray_ref_;
    std::vector<cv::KeyPoint> kp_ref_;
    cv::Mat descriptors_ref_;

    cv::Ptr<cv::Feature2D> feature_extractor_;
    cv::Ptr<cv::Feature2D> descriptor_extractor_;
    cv::Ptr<cv::FastFeatureDetector> fast_detector_;
    int norm_type_ = cv::NORM_HAMMING;

    cv::Mat scale_matrix_;
    cv::Mat scale_matrix_inv_;
    std::vector<cv::Point2f> frame_corners_;
    float frame_area_ = 0.0f;

    std::string last_fail_reason_;
    int last_match_count_ = 0;
    int last_inlier_count_ = 0;
    float last_overlap_ = 0.0f;
    std::string last_reference_reason_;
    int re_registration_calls_ = 0;
    bool pending_bg_reset_ = false;

    cv::cuda::GpuMat frame_gpu_;
    cv::cuda::GpuMat reg_gpu_;

    cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> backsub_;
    cv::Ptr<cv::cuda::Filter> gaussian_filter_;
    cv::Mat morph_kernel_;
    int close_iterations_ = 1;
    int open_iterations_ = 1;
    int dilate_iterations_ = 2;
    int min_contour_area_ = 50;
    int max_contour_area_ = 1500;
    float border_gate_percent_ = 1.0f;

    cv::cuda::GpuMat frame_small_gpu_;
    cv::cuda::GpuMat blur_small_gpu_;
    cv::cuda::GpuMat fg_gpu_;
    cv::cuda::GpuMat mask_gpu_;

    cv::Mat mask_cpu_;
    cv::Mat cc_mask_cpu_;
    cv::Mat bb_mask_cpu_;
    cv::Mat registered_cpu_;
    std::vector<int> reference_events_;
    bool return_registered_ = true;
};
