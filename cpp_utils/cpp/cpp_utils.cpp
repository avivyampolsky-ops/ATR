#include "cpp_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace fs = std::filesystem;

namespace cpp_utils {

namespace {

std::string trim(const std::string &value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        start++;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        end--;
    }
    return value.substr(start, end - start);
}

std::string strip_comment(const std::string &line) {
    size_t pos = line.find('#');
    if (pos == std::string::npos) {
        return line;
    }
    return line.substr(0, pos);
}

bool parse_bool(const std::string &value, bool fallback) {
    std::string lower;
    lower.reserve(value.size());
    for (char c : value) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "true" || lower == "yes" || lower == "on") {
        return true;
    }
    if (lower == "false" || lower == "no" || lower == "off") {
        return false;
    }
    return fallback;
}

int parse_int(const std::string &value, int fallback) {
    try {
        size_t idx = 0;
        int out = std::stoi(value, &idx);
        if (idx == value.size()) {
            return out;
        }
    } catch (...) {
    }
    return fallback;
}

float parse_float(const std::string &value, float fallback) {
    try {
        size_t idx = 0;
        float out = std::stof(value, &idx);
        if (idx == value.size()) {
            return out;
        }
    } catch (...) {
    }
    return fallback;
}

bool is_image_file(const fs::path &path) {
    if (!path.has_extension()) {
        return false;
    }
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

bool natural_less(const std::string &a, const std::string &b) {
    size_t ia = 0;
    size_t ib = 0;
    while (ia < a.size() && ib < b.size()) {
        bool da = std::isdigit(static_cast<unsigned char>(a[ia]));
        bool db = std::isdigit(static_cast<unsigned char>(b[ib]));
        if (da && db) {
            size_t ja = ia;
            size_t jb = ib;
            while (ja < a.size() && std::isdigit(static_cast<unsigned char>(a[ja]))) {
                ja++;
            }
            while (jb < b.size() && std::isdigit(static_cast<unsigned char>(b[jb]))) {
                jb++;
            }
            std::string sa = a.substr(ia, ja - ia);
            std::string sb = b.substr(ib, jb - ib);
            long long na = std::stoll(sa);
            long long nb = std::stoll(sb);
            if (na != nb) {
                return na < nb;
            }
            if (sa.size() != sb.size()) {
                return sa.size() < sb.size();
            }
            ia = ja;
            ib = jb;
        } else {
            char ca = static_cast<char>(std::tolower(static_cast<unsigned char>(a[ia])));
            char cb = static_cast<char>(std::tolower(static_cast<unsigned char>(b[ib])));
            if (ca != cb) {
                return ca < cb;
            }
            ia++;
            ib++;
        }
    }
    return a.size() < b.size();
}

}  // namespace

bool Config::load(const std::string &path) {
    values.clear();
    runtime.clear();
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Config file not found: " << path << "\n";
        return false;
    }
    std::vector<std::string> key_stack;
    std::string line;
    while (std::getline(in, line)) {
        std::string raw = strip_comment(line);
        if (raw.empty()) {
            continue;
        }
        size_t indent = 0;
        while (indent < raw.size() && raw[indent] == ' ') {
            indent++;
        }
        std::string content = trim(raw);
        if (content.empty()) {
            continue;
        }
        size_t colon = content.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        std::string key = trim(content.substr(0, colon));
        std::string value = trim(content.substr(colon + 1));
        size_t level = indent / 2;
        if (key_stack.size() > level) {
            key_stack.resize(level);
        }
        if (value.empty()) {
            if (key_stack.size() == level) {
                key_stack.push_back(key);
            } else if (key_stack.size() > level) {
                key_stack[level] = key;
            }
            continue;
        }
        std::string full_key;
        for (size_t i = 0; i < key_stack.size(); ++i) {
            if (!full_key.empty()) {
                full_key.push_back('.');
            }
            full_key += key_stack[i];
        }
        if (!full_key.empty()) {
            full_key.push_back('.');
        }
        full_key += key;
        values[full_key] = value;
    }
    return true;
}

void Config::set_runtime(const std::string &key, const std::string &value) {
    runtime[key] = value;
}

void Config::set_runtime(const std::string &key, int value) {
    runtime[key] = std::to_string(value);
}

void Config::set_runtime(const std::string &key, float value) {
    std::ostringstream oss;
    oss << value;
    runtime[key] = oss.str();
}

std::string Config::get_string(const std::string &key, const std::string &fallback) const {
    auto it_runtime = runtime.find(key);
    if (it_runtime != runtime.end()) {
        return it_runtime->second;
    }
    auto it = values.find(key);
    if (it == values.end()) {
        return fallback;
    }
    return it->second;
}

bool Config::get_bool(const std::string &key, bool fallback) const {
    return parse_bool(get_string(key, fallback ? "true" : "false"), fallback);
}

int Config::get_int(const std::string &key, int fallback) const {
    return parse_int(get_string(key, std::to_string(fallback)), fallback);
}

float Config::get_float(const std::string &key, float fallback) const {
    return parse_float(get_string(key, std::to_string(fallback)), fallback);
}

bool DataStreamer::init_source() {
    if (fs::is_directory(data_path)) {
        image_paths.clear();
        for (const auto &entry : fs::directory_iterator(data_path)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            if (is_image_file(entry.path())) {
                image_paths.push_back(entry.path().string());
            }
        }
        std::sort(image_paths.begin(), image_paths.end(), [](const std::string &lhs, const std::string &rhs) {
            return natural_less(fs::path(lhs).filename().string(), fs::path(rhs).filename().string());
        });
        if (image_paths.empty()) {
            std::cerr << "No images found in " << data_path << "\n";
            return false;
        }
        total_frames = static_cast<int>(image_paths.size());
        if (start_frame >= total_frames.value()) {
            std::cerr << "start_frame " << start_frame << " exceeds total frames " << total_frames.value() << "\n";
            return false;
        }
        if (end_frame.has_value()) {
            end_frame = std::min(end_frame.value(), total_frames.value() - 1);
        }
        ref = read_image(image_paths[start_frame]);
        if (ref.empty()) {
            std::cerr << "Failed to read image: " << image_paths[start_frame] << "\n";
            return false;
        }
        resolved_input_is_gray = (ref.channels() == 1);
        fps = fps_override;
        return true;
    }

    cap.open(data_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << data_path << "\n";
        return false;
    }
    int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (total > 0) {
        total_frames = total;
    }
    if (start_frame > 0) {
        cap.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(start_frame));
    }
    if (total_frames.has_value()) {
        if (start_frame >= total_frames.value()) {
            std::cerr << "start_frame " << start_frame << " exceeds total frames " << total_frames.value() << "\n";
            cap.release();
            return false;
        }
        if (end_frame.has_value()) {
            end_frame = std::min(end_frame.value(), total_frames.value() - 1);
        }
    }
    bool ret = cap.read(ref);
    if (!ret || ref.empty()) {
        std::cerr << "Failed to load video reference frame\n";
        cap.release();
        return false;
    }
    if (input_is_gray && ref.channels() != 1) {
        cv::cvtColor(ref, ref, cv::COLOR_BGR2GRAY);
    }
    resolved_input_is_gray = (ref.channels() == 1);
    fps = static_cast<float>(cap.get(cv::CAP_PROP_FPS));
    if (fps <= 0.0f) {
        fps = fps_override;
    }
    return true;
}

cv::Mat DataStreamer::read_image(const std::string &path) const {
    int flag = input_is_gray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
    return cv::imread(path, flag);
}

cv::Mat DataStreamer::get_frame(int frame_idx) {
    if (!image_paths.empty()) {
        if (end_frame.has_value() && frame_idx > end_frame.value()) {
            return cv::Mat();
        }
        if (frame_idx >= static_cast<int>(image_paths.size())) {
            return cv::Mat();
        }
        cv::Mat frame = read_image(image_paths[frame_idx]);
        if (frame.empty()) {
            std::cerr << "Failed to read image: " << image_paths[frame_idx] << "\n";
            return cv::Mat();
        }
        return frame;
    }

    if (!cap.isOpened()) {
        return cv::Mat();
    }
    cv::Mat frame;
    bool ret = cap.read(frame);
    if (!ret || frame.empty()) {
        return cv::Mat();
    }
    if (input_is_gray && frame.channels() != 1) {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    }
    if (end_frame.has_value() && frame_idx > end_frame.value()) {
        return cv::Mat();
    }
    if (total_frames.has_value() && frame_idx >= total_frames.value()) {
        return cv::Mat();
    }
    return frame;
}

void DataStreamer::close() {
    if (cap.isOpened()) {
        cap.release();
    }
}

DebugUtilsCpp::DebugUtilsCpp(const Config &config,
                             const std::string &data_path,
                             const std::string &out_root,
                             float stream_fps_in,
                             bool stream_is_gray_in)
    : registration_mode(config.get_string("registration.mode", "translation")),
      start_frame(std::max(0, config.get_int("debug.start_frame", 0))),
      draw_output(config.get_bool("debug.draw_output", true)),
      debug_mode(config.get_bool("debug.debug_mode", true)),
      enable_timing(config.get_bool("debug.enable_timing", true)),
      write_empty_masks(config.get_bool("debug.write_empty_masks", false)),
      track_min_age(config.get_int("tracker.kalman.track_min_age", 3)),
      incrimination_thresh(config.get_int("tracker.kalman.incrimination_thresh", -1)),
      stream_fps(stream_fps_in),
      stream_is_gray(stream_is_gray_in) {
    int end = config.get_int("debug.end_frame", -1);
    if (end >= 0) {
        end_frame = end;
    }
    fs::path base_path(data_path);
    video_base = base_path.filename().string();
    if (video_base.empty()) {
        video_base = "input";
    }
    size_t dot = video_base.find_last_of('.');
    if (dot != std::string::npos) {
        video_base = video_base.substr(0, dot);
    }
    out_dir = (fs::path(out_root) / video_base).string();
    frames_dir = (fs::path(out_dir) / ("frames_" + video_base)).string();
    mog2_dir = (fs::path(out_dir) / ("mog2_masks_" + video_base)).string();
    bb_dir = (fs::path(out_dir) / ("bb_masks_" + video_base)).string();

    write_empty_masks = write_empty_masks || draw_output;

    if (debug_mode) {
        fs::create_directories(out_dir);
        if (draw_output) {
            fs::create_directories(frames_dir);
            fs::create_directories(mog2_dir);
            fs::create_directories(bb_dir);
        }
        reanchor_log_path = (fs::path(out_dir) / ("reanchor_log_" + video_base + ".txt")).string();
        if (incrimination_thresh >= 0) {
            std::string handshake_path = (fs::path(out_dir) / "vmd_handshake.csv").string();
            handshake_log.open(handshake_path);
            if (handshake_log.is_open()) {
                handshake_log << "frame_input,frame_proc,track_id,age,x,y,w,h\n";
                handshake_log.flush();
            }
        }
    }
    if (debug_mode && registration_mode == "homography") {
        homography_min_inliers = config.get_int("registration.homography.min_inliers", 0);
        std::string log_path = (fs::path(out_dir) / ("homography_log_" + video_base + ".txt")).string();
        homography_log.open(log_path);
        if (homography_log.is_open()) {
            homography_log << "frame,match_count,inlier_count,overlap_ratio,fail_reason,min_inliers\n";
            homography_log.flush();
        }
    }
}

void DebugUtilsCpp::set_stream_info(const DataStreamer &streamer) {
    total_frames = streamer.total_frames;
    start_frame = streamer.start_frame;
    end_frame = streamer.end_frame;
}

void DebugUtilsCpp::close() {
    if (!debug_mode) {
        return;
    }
    if (writer.isOpened()) {
        writer.release();
    }
    if (homography_log.is_open()) {
        homography_log.close();
    }
    if (handshake_log.is_open()) {
        handshake_log.close();
    }
}

void DebugUtilsCpp::write_homography_log(int frame_idx, const HomographyRegistrationDebug *homography_debug) {
    if (!debug_mode || !homography_log.is_open() || homography_debug == nullptr) {
        return;
    }
    homography_log << frame_idx << ","
                   << homography_debug->match_count << ","
                   << homography_debug->inlier_count << ","
                   << homography_debug->overlap_ratio << ","
                   << (homography_debug->fail_reason.empty() ? "ok" : homography_debug->fail_reason) << ","
                   << homography_min_inliers << "\n";
    homography_log.flush();
}

void DebugUtilsCpp::set_shape(const cv::Size &shape) {
    if (shape.width > 0 && shape.height > 0) {
        registration_margin = static_cast<int>(std::min(shape.width, shape.height) * 0.01f);
    }
    if (!debug_mode || !draw_output) {
        return;
    }
    if (!writer.isOpened() && stream_fps > 0.0f) {
        writer_path = (fs::path(out_dir) / ("tracked_" + video_base + ".mp4")).string();
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(writer_path,
                    fourcc,
                    stream_fps,
                    shape,
                    !stream_is_gray);
        if (!writer.isOpened()) {
            writer.release();
            writer_path = (fs::path(out_dir) / ("tracked_" + video_base + ".avi")).string();
            fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            writer.open(writer_path,
                        fourcc,
                        stream_fps,
                        shape,
                        !stream_is_gray);
        }
        if (!writer.isOpened() && !writer_warned) {
            writer_warned = true;
            std::cerr << "Warning: failed to open video writer for debug output.\n";
        }
        if (writer.isOpened()) {
            writer_size = shape;
            writer_is_color = !stream_is_gray;
        }
    }
}

void DebugUtilsCpp::note_loop_call() {
    loop_calls += 1;
}

void DebugUtilsCpp::write_reference_events_graph(const std::vector<int> &ref_events) {
    if (!debug_mode) {
        return;
    }
    if (ref_events.empty() || loop_calls <= 0) {
        return;
    }
    int width = std::min(2000, loop_calls);
    int height = 200;
    cv::Mat graph(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    int span = std::max(1, loop_calls - 1);
    for (int ev : ref_events) {
        int frame_global = start_frame + ev;
        if (end_frame.has_value() && frame_global > end_frame.value()) {
            continue;
        }
        int rel = frame_global - start_frame;
        if (rel < 0 || rel >= loop_calls) {
            continue;
        }
        int x = static_cast<int>(std::round((static_cast<double>(rel) / span) * (width - 1)));
        cv::line(graph, cv::Point(x, 0), cv::Point(x, height - 1), cv::Scalar(0, 0, 255), 1);
    }
    cv::imwrite((fs::path(out_dir) / ("reference_events_" + video_base + ".png")).string(), graph);
}

cv::Mat DebugUtilsCpp::annotate_registration_debug(const cv::Mat &frame,
                                                   int frame_idx,
                                                   const std::set<int> &ref_events,
                                                   const HomographyRegistrationDebug *homography_debug,
                                                   const cv::Point2f &shift) {
    if (!debug_mode || frame.empty()) {
        return frame;
    }
    cv::Mat out = frame.clone();
    if (!ref_events.empty() && ref_events.count(frame_idx)) {
        std::string label = "REANCHOR";
        if (registration_mode == "homography" && homography_debug != nullptr) {
            if (!homography_debug->reference_reason.empty()) {
                label = "REANCHOR " + homography_debug->reference_reason;
            }
        }
        cv::Scalar color = out.channels() == 1 ? cv::Scalar(255) : cv::Scalar(0, 0, 255);
        cv::putText(out, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
    }
    if (registration_mode == "homography" && homography_debug != nullptr) {
        std::string reason = homography_debug->fail_reason.empty() ? "ok" : homography_debug->fail_reason;
        int matches = homography_debug->match_count;
        int inliers = homography_debug->inlier_count;
        double overlap = homography_debug->overlap_ratio;
        cv::Scalar color = out.channels() == 1 ? cv::Scalar(255) : cv::Scalar(0, 255, 255);
        std::ostringstream oss;
        oss << "HOMO " << reason << " m=" << matches << " i=" << inliers << " ov="
            << std::fixed << std::setprecision(2) << overlap;
        cv::putText(out, oss.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    } else {
        cv::Scalar color = out.channels() == 1 ? cv::Scalar(255) : cv::Scalar(0, 255, 255);
        std::ostringstream oss;
        oss << "SHIFT dx=" << std::fixed << std::setprecision(2) << shift.x
            << " dy=" << std::fixed << std::setprecision(2) << shift.y;
        cv::putText(out, oss.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
    return out;
}

cv::Mat DebugUtilsCpp::draw_tracks(const cv::Mat &frame,
                                   const std::map<int, KalmanIoUTracker::Entity> &tracks) const {
    if (!debug_mode || frame.empty()) {
        return frame;
    }
    cv::Mat out = frame.clone();
    cv::Scalar default_color = out.channels() == 1 ? cv::Scalar(255) : cv::Scalar(0, 255, 0);
    cv::Scalar incriminated_color = out.channels() == 1 ? cv::Scalar(255) : cv::Scalar(0, 0, 255);
    cv::Scalar label_color = out.channels() == 1 ? cv::Scalar(255) : cv::Scalar(0, 0, 255);
    for (const auto &kv : tracks) {
        const auto &entity = kv.second;
        if (!entity.moving) {
            continue;
        }
        cv::Scalar color;
        if (incrimination_thresh >= 0) {
            int green_age = (3 * incrimination_thresh + 3) / 4;
            if (entity.age < green_age) {
                continue;
            }
            color = (entity.age >= incrimination_thresh) ? incriminated_color : default_color;
        } else {
            if (entity.age < track_min_age) {
                continue;
            }
            color = default_color;
        }
        int x = static_cast<int>(std::round(entity.bbox.x));
        int y = static_cast<int>(std::round(entity.bbox.y));
        int w = static_cast<int>(std::round(entity.bbox.w));
        int h = static_cast<int>(std::round(entity.bbox.h));
        cv::rectangle(out, cv::Rect(x, y, w, h), color, 2);
        std::ostringstream oss;
        oss << "ID " << entity.id << " AGE:" << entity.age;
        cv::putText(out, oss.str(), cv::Point(x, y - 6), cv::FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2);
    }
    return out;
}

void DebugUtilsCpp::write_debug_images(const cv::Mat &frame,
                                       int frame_idx,
                                       const cv::Mat &mask,
                                       const cv::Mat &cc_mask,
                                       const std::vector<cv::Rect> &detections) const {
    if (!debug_mode) {
        return;
    }
    if (!frame.empty()) {
        cv::imwrite((fs::path(frames_dir) / ("frame_" + pad_index(frame_idx) + ".jpg")).string(), frame);
    }
    cv::Mat mask_to_write = mask;
    if (mask_to_write.empty() && write_empty_masks) {
        cv::Size mask_size;
        if (!frame.empty()) {
            mask_size = frame.size();
        } else if (!cc_mask.empty()) {
            mask_size = cc_mask.size();
        } else if (writer_size.width > 0 && writer_size.height > 0) {
            mask_size = writer_size;
        }
        if (mask_size.width > 0 && mask_size.height > 0) {
            mask_to_write = cv::Mat::zeros(mask_size, CV_8UC1);
        }
    }
    if (!mask_to_write.empty()) {
        cv::imwrite((fs::path(mog2_dir) / ("mask_" + pad_index(frame_idx) + ".jpg")).string(), mask_to_write);
        cv::Mat bb_mask;
        if (!cc_mask.empty()) {
            bb_mask = cc_mask;
            if (bb_mask.channels() == 1) {
                cv::cvtColor(bb_mask, bb_mask, cv::COLOR_GRAY2BGR);
            } else {
                bb_mask = bb_mask.clone();
            }
        } else {
            cv::Size bb_size;
            if (!frame.empty()) {
                bb_size = frame.size();
            } else if (mask_to_write.size().width > 0 && mask_to_write.size().height > 0) {
                bb_size = mask_to_write.size();
            } else if (writer_size.width > 0 && writer_size.height > 0) {
                bb_size = writer_size;
            }
            if (bb_size.width > 0 && bb_size.height > 0) {
                bb_mask = cv::Mat::zeros(bb_size, CV_8UC3);
            }
        }
        if (!bb_mask.empty()) {
            for (const auto &rect : detections) {
                cv::rectangle(bb_mask, rect, cv::Scalar(0, 0, 255), 2);
            }
            cv::imwrite((fs::path(bb_dir) / ("mask_" + pad_index(frame_idx) + ".jpg")).string(), bb_mask);
        }
    }
}

void DebugUtilsCpp::process_image_outputs(const cv::Mat &registered_frame,
                                          int frame_idx,
                                          int frame_count,
                                          const std::set<int> &ref_events,
                                          const HomographyRegistrationDebug *homography_debug,
                                          const cv::Point2f &shift,
                                          const std::map<int, KalmanIoUTracker::Entity> &tracks,
                                          const cv::Mat &mask,
                                          const cv::Mat &cc_mask,
                                          const std::vector<cv::Rect> &detections) {
    write_handshake(frame_idx, frame_count, shift, tracks);
    if (!debug_mode || !draw_output) {
        return;
    }
    cv::Mat out = annotate_registration_debug(registered_frame, frame_count, ref_events, homography_debug, shift);
    out = draw_tracks(out, tracks);
    write_debug_images(out, frame_idx, mask, cc_mask, detections);
    if (writer.isOpened()) {
        cv::Mat frame_to_write = out;
        if (frame_to_write.empty()) {
            return;
        }
        if (frame_to_write.size() != writer_size) {
            cv::resize(frame_to_write, frame_to_write, writer_size, 0.0, 0.0, cv::INTER_NEAREST);
        }
        if (writer_is_color && frame_to_write.channels() == 1) {
            cv::cvtColor(frame_to_write, frame_to_write, cv::COLOR_GRAY2BGR);
        } else if (!writer_is_color && frame_to_write.channels() == 3) {
            cv::cvtColor(frame_to_write, frame_to_write, cv::COLOR_BGR2GRAY);
        }
        writer.write(frame_to_write);
    } else if (!writer_warned) {
        writer_warned = true;
        std::cerr << "Warning: debug video writer not open; skipping frame writes.\n";
    }
}

void DebugUtilsCpp::write_handshake(int frame_idx,
                                    int frame_count,
                                    const cv::Point2f &shift,
                                    const std::map<int, KalmanIoUTracker::Entity> &tracks) {
    if (!handshake_log.is_open() || incrimination_thresh < 0) {
        return;
    }
    for (const auto &kv : tracks) {
        const auto &entity = kv.second;
        if (entity.age != incrimination_thresh) {
            continue;
        }
        int x = static_cast<int>(std::round(entity.bbox.x));
        int y = static_cast<int>(std::round(entity.bbox.y));
        int w = static_cast<int>(std::round(entity.bbox.w));
        int h = static_cast<int>(std::round(entity.bbox.h));
        if (registration_mode == "translation") {
            x = static_cast<int>(std::round(static_cast<float>(x) + static_cast<float>(registration_margin) - shift.x));
            y = static_cast<int>(std::round(static_cast<float>(y) + static_cast<float>(registration_margin) - shift.y));
        }
        handshake_log << frame_idx << ","
                      << frame_count << ","
                      << entity.id << ","
                      << entity.age << ","
                      << x << "," << y << "," << w << "," << h << "\n";
        handshake_log.flush();
    }
}

void DebugUtilsCpp::build_frame_vs_mask_video(float fps) {
    if (!debug_mode || !draw_output) {
        return;
    }
    if (!fs::exists(frames_dir) || !fs::exists(bb_dir)) {
        std::cerr << "Missing frames or bb_masks folder.\n";
        return;
    }
    std::vector<std::string> frame_paths;
    std::vector<std::string> mask_paths;
    for (const auto &entry : fs::directory_iterator(frames_dir)) {
        if (entry.is_regular_file() && is_image_file(entry.path())) {
            frame_paths.push_back(entry.path().string());
        }
    }
    for (const auto &entry : fs::directory_iterator(bb_dir)) {
        if (entry.is_regular_file() && is_image_file(entry.path())) {
            mask_paths.push_back(entry.path().string());
        }
    }
    if (frame_paths.empty() || mask_paths.empty()) {
        std::cerr << "No frames or masks found.\n";
        return;
    }
    std::sort(frame_paths.begin(), frame_paths.end(), [](const std::string &lhs, const std::string &rhs) {
        return natural_less(fs::path(lhs).filename().string(), fs::path(rhs).filename().string());
    });
    std::sort(mask_paths.begin(), mask_paths.end(), [](const std::string &lhs, const std::string &rhs) {
        return natural_less(fs::path(lhs).filename().string(), fs::path(rhs).filename().string());
    });
    cv::Mat first = cv::imread(frame_paths.front(), cv::IMREAD_COLOR);
    if (first.empty()) {
        std::cerr << "Failed to read first frame.\n";
        return;
    }
    int h = first.rows;
    int w = first.cols;
    std::string out_path = (fs::path(out_dir) / ("side_by_side_" + video_base + ".mp4")).string();
    cv::VideoWriter out(out_path,
                        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        fps,
                        cv::Size(w * 2, h));
    if (!out.isOpened()) {
        out.release();
        out_path = (fs::path(out_dir) / ("side_by_side_" + video_base + ".avi")).string();
        out.open(out_path,
                 cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                 fps,
                 cv::Size(w * 2, h));
        if (!out.isOpened()) {
            std::cerr << "Failed to open side-by-side writer.\n";
            return;
        }
    }

    auto extract_index = [](const std::string &path) -> std::optional<int> {
        std::string name = fs::path(path).filename().string();
        size_t pos = name.find_first_of("0123456789");
        if (pos == std::string::npos) {
            return std::nullopt;
        }
        size_t end = pos;
        while (end < name.size() && std::isdigit(static_cast<unsigned char>(name[end]))) {
            end++;
        }
        try {
            return std::stoi(name.substr(pos, end - pos));
        } catch (...) {
            return std::nullopt;
        }
    };

    std::map<int, std::string> frame_map;
    std::map<int, std::string> mask_map;
    for (const auto &path : frame_paths) {
        auto idx = extract_index(path);
        if (idx.has_value()) {
            frame_map[idx.value()] = path;
        }
    }
    for (const auto &path : mask_paths) {
        auto idx = extract_index(path);
        if (idx.has_value()) {
            mask_map[idx.value()] = path;
        }
    }
    if (frame_map.empty() || mask_map.empty()) {
        std::cerr << "Missing frame or mask indices.\n";
        return;
    }
    for (const auto &kv : frame_map) {
        int frame_idx = kv.first;
        auto mask_it = mask_map.find(frame_idx);
        if (mask_it == mask_map.end()) {
            continue;
        }
        cv::Mat frame = cv::imread(kv.second, cv::IMREAD_COLOR);
        cv::Mat mask = cv::imread(mask_it->second, cv::IMREAD_COLOR);
        if (frame.empty() || mask.empty()) {
            continue;
        }
        if (frame.size() != mask.size()) {
            cv::resize(mask, mask, frame.size(), 0.0, 0.0, cv::INTER_NEAREST);
        }
        cv::Mat combined;
        cv::hconcat(frame, mask, combined);
        out.write(combined);
    }
}

void DebugUtilsCpp::print_run_summary(const TimingSummary &reg_summary,
                                      const TimingSummary &det_summary,
                                      const TimingSummary &tracker_summary,
                                      double loop_total,
                                      float fps) {
    if (!debug_mode) {
        return;
    }
    std::vector<std::string> lines;
    append_registrator_summary(lines, reg_summary);
    append_detector_summary(lines, det_summary);
    append_tracker_summary(lines, tracker_summary);
    if (loop_calls > 0) {
        std::ostringstream oss;
        oss << "Loop avg (ms/frame): " << std::fixed << std::setprecision(3)
            << (loop_total / loop_calls) * 1000.0;
        lines.push_back(oss.str());
    }
    if (draw_output) {
        build_frame_vs_mask_video(fps);
    }
    lines.push_back("Done");
    for (const auto &line : lines) {
        std::cout << line << "\n";
    }
    if (enable_timing && !lines.empty()) {
        std::string timing_path = (fs::path(out_dir) / ("timing_summary_" + video_base + ".txt")).string();
        std::ofstream log_file(timing_path);
        if (log_file.is_open()) {
            for (const auto &line : lines) {
                log_file << line << "\n";
            }
        }
    }
}

std::string DebugUtilsCpp::pad_index(int idx) {
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << idx;
    return oss.str();
}

void DebugUtilsCpp::append_registrator_summary(std::vector<std::string> &lines, const TimingSummary &summary) {
    if (!summary.enabled) {
        return;
    }
    lines.push_back("Registrator summary (avg ms/frame):");
    if (summary.calls == 0) {
        lines.push_back("  no calls recorded");
        return;
    }
    lines.push_back("  calls: " + std::to_string(summary.calls));
    append_value(lines, summary, "gray_ms_avg", "  gray: ");
    append_value(lines, summary, "keypoints_ms_avg", "  keypoints: ");
    append_value(lines, summary, "match_ms_avg", "  match: ");
    append_value(lines, summary, "homography_ms_avg", "  homography: ");
    append_value(lines, summary, "shift_ms_avg", "  shift: ");
    append_value(lines, summary, "warp_ms_avg", "  warp: ");
    append_value(lines, summary, "upload_ms_avg", "  upload: ");

    double reg_sum = std::numeric_limits<double>::quiet_NaN();
    if (has_value(summary, "registration_sum_ms_avg")) {
        reg_sum = summary.values.at("registration_sum_ms_avg");
    } else if (has_value(summary, "registration_ms_avg")) {
        reg_sum = summary.values.at("registration_ms_avg");
    } else if (has_value(summary, "registration")) {
        reg_sum = summary.values.at("registration");
    }
    if (!std::isnan(reg_sum)) {
        std::ostringstream oss;
        oss << "  registration sum: " << std::fixed << std::setprecision(3) << reg_sum;
        lines.push_back(oss.str());
    }
    if (has_value(summary, "re_registration_calls")) {
        lines.push_back("  re-registrations: " + to_string(summary, "re_registration_calls"));
    }
    append_value(lines, summary, "registration_first_try_ms_avg", "  registration (first try): ");
    append_value(lines, summary, "registration_fallback_ms_avg", "  registration (fallback): ");
    append_value(lines, summary, "phase_response_avg", "  phase response avg: ");
    append_value(lines, summary, "ecc_cc_avg", "  ecc cc avg: ");
}

void DebugUtilsCpp::append_detector_summary(std::vector<std::string> &lines, const TimingSummary &summary) {
    if (!summary.enabled) {
        return;
    }
    lines.push_back("Detector summary (avg ms/frame):");
    if (summary.calls == 0) {
        lines.push_back("  no calls recorded");
        return;
    }
    lines.push_back("  calls: " + std::to_string(summary.calls));
    append_value(lines, summary, "blur_ms_avg", "  blur: ");
    append_value(lines, summary, "gray_ms_avg", "  gray: ");
    append_value(lines, summary, "diff_ms_avg", "  diff: ");
    append_value(lines, summary, "bgsub_ms_avg", "  bgsub: ");
    append_value(lines, summary, "contours_ms_avg", "  contours: ");
    append_value(lines, summary, "bbox_creation_ms_avg", "  bbox_creation: ");
    append_value(lines, summary, "download_ms_avg", "  download: ");
    double det_sum = std::numeric_limits<double>::quiet_NaN();
    if (has_value(summary, "detection_sum_ms_avg")) {
        det_sum = summary.values.at("detection_sum_ms_avg");
    } else if (has_value(summary, "total_ms_avg")) {
        det_sum = summary.values.at("total_ms_avg");
    }
    if (!std::isnan(det_sum)) {
        std::ostringstream oss;
        oss << "  detection sum: " << std::fixed << std::setprecision(3) << det_sum;
        lines.push_back(oss.str());
    }
}

void DebugUtilsCpp::append_tracker_summary(std::vector<std::string> &lines, const TimingSummary &summary) {
    if (!summary.enabled) {
        return;
    }
    lines.push_back("KalmanIoUTracker summary (avg ms/frame):");
    if (summary.calls == 0) {
        lines.push_back("  no calls recorded");
        return;
    }
    lines.push_back("  calls: " + std::to_string(summary.calls));
    append_value(lines, summary, "update_ms_avg", "  update: ");
    append_value(lines, summary, "update_ms_avg", "  total: ");
}

bool DebugUtilsCpp::has_value(const TimingSummary &summary, const std::string &key) {
    return summary.values.find(key) != summary.values.end();
}

std::string DebugUtilsCpp::to_string(const TimingSummary &summary, const std::string &key) {
    auto it = summary.values.find(key);
    if (it == summary.values.end()) {
        return "0";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << it->second;
    return oss.str();
}

void DebugUtilsCpp::append_value(std::vector<std::string> &lines,
                                 const TimingSummary &summary,
                                 const std::string &key,
                                 const std::string &label) {
    auto it = summary.values.find(key);
    if (it == summary.values.end()) {
        return;
    }
    std::ostringstream oss;
    oss << label << std::fixed << std::setprecision(3) << it->second;
    lines.push_back(oss.str());
}

int compute_total_frames(const DataStreamer &streamer) {
    int start = streamer.start_frame;
    if (streamer.end_frame.has_value()) {
        int end = streamer.end_frame.value();
        return std::max(0, end - start + 1);
    }
    if (streamer.total_frames.has_value()) {
        return std::max(0, streamer.total_frames.value() - start);
    }
    return -1;
}

}  // namespace cpp_utils
