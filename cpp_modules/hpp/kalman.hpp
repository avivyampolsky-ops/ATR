#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <map>
#include <vector>

struct BBox {
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
};

class KalmanIoUTracker {
public:
    struct Entity {
        int id = 0;
        BBox bbox;
        cv::KalmanFilter kf;
        int age = 1;
        int lost = 0;
        bool moving = false;
        float ema_area = 0.0f;
    };

    KalmanIoUTracker(
        float iou_thresh = 0.05f,
        int max_lost = 8,
        float min_move = 2.0f,
        float ema_alpha = 0.3f,
        float dist_gate_scale = 2.0f,
        float easy_iou_thresh = 0.6f,
        float area_gate_min_scale = 0.5f,
        float area_gate_max_scale = 2.0f
    );

    void apply_shift(float dx, float dy);
    void reset();
    const std::map<int, Entity> &get_tracks() const;
    std::map<int, Entity> update(const std::vector<cv::Rect> &observations);

private:
    static void init_kalman_state(cv::KalmanFilter &kf, const BBox &bbox);
    static cv::KalmanFilter create_kalman();
    static float iou(const BBox &a, const BBox &b);

    std::vector<std::vector<float>> build_iou_matrix(
        const std::vector<std::pair<int, Entity>> &track_items,
        const std::vector<BBox> &obs
    ) const;

    void apply_match(Entity &entity, const BBox &bbox);

    float iou_thresh_ = 0.05f;
    int max_lost_ = 8;
    float min_move_ = 2.0f;
    float ema_alpha_ = 0.3f;
    float dist_gate_scale_ = 2.0f;
    float easy_iou_thresh_ = 0.6f;
    float area_gate_min_scale_ = 0.5f;
    float area_gate_max_scale_ = 2.0f;

    int current_id_ = 0;
    std::map<int, Entity> tracks_;
};
