#include <Optimium/Runtime.h>
#include <Optimium/Runtime/ToString.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>
#include <iostream>
#include <cmath>

namespace fs = std::filesystem;
namespace rt = optimium::runtime;

using timer = std::chrono::high_resolution_clock;

constexpr size_t kNumLandmarks = 39;
const static std::array<cv::Point, 35> Vertexes = {
    cv::Point{0, 1},
    cv::Point{1, 2},
    cv::Point{2, 3},
    cv::Point{3, 7},
    cv::Point{0, 4},
    cv::Point{4, 5},
    cv::Point{5, 6},
    cv::Point{6, 8},
    cv::Point{9, 10},
    cv::Point{11, 12},
    cv::Point{11, 13},
    cv::Point{13, 15},
    cv::Point{15, 17},
    cv::Point{17, 19},
    cv::Point{19, 15},
    cv::Point{15, 21},
    cv::Point{12, 14},
    cv::Point{14, 16},
    cv::Point{16, 18},
    cv::Point{16, 22},
    cv::Point{18, 20},
    cv::Point{20, 16},
    cv::Point{12, 24},
    cv::Point{11, 23},
    cv::Point{24, 23},
    cv::Point{24, 26},
    cv::Point{26, 28},
    cv::Point{28, 32},
    cv::Point{28, 30},
    cv::Point{32, 30},
    cv::Point{23, 25},
    cv::Point{25, 27},
    cv::Point{27, 29},
    cv::Point{27, 31},
    cv::Point{29, 31}
};

struct Landmark {
    float X;
    float Y;
    float Z;
    float Visibility;
    float Presense;

    cv::Point to_point() const {
        return cv::Point(static_cast<int>(X), static_cast<int>(Y));
    }
};

float sigmoid(float x) {
    return (1.0f / (1.0f + std::exp(-x)));
}

constexpr float kWidth = 640.0f;
constexpr float kHeight = 480.0f;

void decode_landmarks(const size_t num_values, const float* raw_data, std::vector<Landmark>& landmarks) {
    const auto num_dimensions = num_values / kNumLandmarks;
    
    for (auto i = 0; i < kNumLandmarks; ++i) {
        Landmark landmark;
        const float* base = raw_data + (i * num_dimensions);

        landmark.X = (base[0] / 256.0f) * kWidth;
        landmark.Y = (base[1] / 256.0f) * kHeight;

        // This part is not used.
        // if (num_dimensions > 2)
        //     landmark.Z = base[2];
        // if (num_dimensions > 3)
        //     landmark.Visibility = sigmoid(base[3]);
        // if (num_dimensions > 4)
        //     landmark.Presense = sigmoid(base[4]);

        landmarks.push_back(landmark);
    }
}

rt::Result<void> do_main() {
    cv::Mat frame;
    cv::Mat resized(cv::Size2i(256, 256), CV_8UC3);
    cv::Mat transformed(cv::Size2i(256, 256), CV_8UC3);

    cv::VideoCapture capture;

    capture.set(cv::CAP_PROP_FRAME_WIDTH, kWidth);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, kHeight);
    capture.set(cv::CAP_PROP_FPS, 30.0);
    capture.open(0, cv::CAP_V4L2);

    if (!capture.isOpened()) {
        return rt::Error(rt::Status::InitFailure, "failed to open camera.");
    }

    rt::Context context = TRY(rt::Context::create());
    auto model = TRY(context.loadModel("pose_detection_lite"));
    auto output_tensor_info = TRY(model.getOutputTensorInfo("Identity"));
    auto output_size_0 = output_tensor_info.TensorShape.getTotalElementCount();

    auto req = TRY(model.createRequest());
    auto input_tensor = TRY(req.getInputTensor("input_1"));

    std::vector<Landmark> landmarks(kNumLandmarks);

    while (true) {
        auto begin = timer::now();
        capture.read(frame);
        auto read_end = timer::now();

        if (frame.empty()) {
            return rt::Error(rt::Status::DeviceError, "frame is empty.");
        }

        cv::resize(frame, resized, resized.size());
        auto resize_end = timer::now();

        cv::cvtColor(resized, transformed, cv::COLOR_BGR2RGB);
        auto color_end = timer::now();

        {
            auto input_buffer = input_tensor.getRawBuffer();
            auto *raw = input_buffer.cast<float>();
            cv::Mat input(256, 256, CV_32FC3, raw);
            transformed.convertTo(input, CV_32FC3, 1.0f / 255.0f, 0);
        }
        auto type_end = timer::now();

        CHECK(req.infer());
        CHECK(req.wait());
        auto infer_end = timer::now();

        {
            auto output_tensor_0 = TRY(req.getOutputTensor("Identity"));
            auto output_buffer = output_tensor_0.getRawBuffer();

            auto* raw = output_buffer.cast<float>();
            decode_landmarks(output_size_0, raw, landmarks);
        }
        auto decode_end = timer::now();

        for (const auto [start, end] : Vertexes) {
            const auto& start_landmark = landmarks[start];
            const auto& end_landmark = landmarks[end];

            cv::line(frame, start_landmark.to_point(), end_landmark.to_point(),
                     CV_RGB(255, 0, 0), 3);
        }

        for (const auto& landmark : landmarks) {
            cv::circle(frame, landmark.to_point(), 7, CV_RGB(0, 0, 255), -1);
        }

        cv::imshow("image", frame);
        auto show_end = timer::now();

        auto key = cv::waitKey(1000 / 30);
        if (key == 'q') {
            break;
        }

        landmarks.clear();

        // std::cout << "read: " << ((read_end - begin).count() / 1000.0f)
        //           << "us, resize: " << ((resize_end - read_end).count() / 1000.0f)
        //           << "us, cvtColor: " << ((color_end - resize_end).count() / 1000.0f)
        //           << "us, cvtType: " << ((type_end - color_end).count() / 1000.0f)
        //           << "us, infer: " << ((infer_end - type_end).count() / 1000.0f)
        //           << "us, decode: " << ((decode_end - infer_end).count() / 1000.0f)
        //           << "us, show: " << ((show_end - decode_end).count() / 1000.0f) << "us "
        //           << "sum: " << ((show_end - begin).count() / 1000.0f) << "us\n";

        std::cout << (1000.0f / ((show_end - begin).count() / 1000000.0f)) << "fps\n";
    }
   
    return rt::Ok();
}

int main(int argc, char** argv) {
    auto res = do_main();

    if (!res.ok()) {
        std::cout << rt::toString(res.error()) << std::endl;
        return 1;
    }

    return 0;
}