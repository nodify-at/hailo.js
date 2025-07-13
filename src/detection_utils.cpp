/**
 * @file detection_utils.cpp
 * @brief Implementation of object detection output parsing utilities

 * @date 2025
 * 
 * This file implements the parsing of Non-Maximum Suppression (NMS) outputs
 * from object detection models running on Hailo NPU.
 */

#include <detection_utils.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <arm_neon.h>
#include <hailo/hailort.hpp>

namespace hailo_addon {
    /**
     * @brief COCO dataset class names
     * 
     * This array maps class IDs (0-80) to human-readable names.
     * Index 0 is "background" which is typically not used in detections.
     */
    static constexpr std::array<std::string_view, 81> COCO_CLASSES = {
        "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    /**
     * @brief Parse NMS output buffer and transform coordinates
     * 
     * This function handles the complex multi-stage coordinate transformation:
     * 
     * Pipeline stages:
     * 1. Original JPEG (1280x720) -> RGB resize (640x360) for preprocessing
     * 2. RGB (640x360) -> Letterboxed square (640x640) for model input
     * 3. Model outputs normalized coordinates (0-1) in 640x640 space
     * 4. We need to reverse this: Model -> RGB -> JPEG coordinates
     * 
     * The letterboxing process adds padding to maintain aspect ratio,
     * which must be accounted for in the coordinate transformation.
     */
    std::vector<Detection> parse_nms_output(
        std::span<const uint8_t> data,
        size_t num_classes,
        float confidence_threshold,
        int jpeg_width,        // Final display size (1280x720)
        int jpeg_height,
        int rgb_width,         // Intermediate RGB size (640x360)
        int rgb_height,
        int model_input_size   // Square inference size (640x640)
    ) {
        std::vector<Detection> detections;
        detections.reserve(200);  // Pre-allocate for typical detection count

        // Step 1: Calculate letterbox parameters for RGB->Model transformation
        // When resizing 640x360 to 640x640 with letterboxing:
        // - Calculate scale to fit the image within the square
        // - Add padding to center the scaled image
        float scale = std::min(
            static_cast<float>(model_input_size) / rgb_width,
            static_cast<float>(model_input_size) / rgb_height
        );

        int scaled_width = static_cast<int>(rgb_width * scale);
        int scaled_height = static_cast<int>(rgb_height * scale);

        // Calculate padding needed to center the image
        float pad_x = (model_input_size - scaled_width) / 2.0f;
        float pad_y = (model_input_size - scaled_height) / 2.0f;

        // Step 2: Calculate scaling factors for coordinate transformations
        float rgb_to_jpeg_scale_x = static_cast<float>(jpeg_width) / rgb_width;   // 1280/640 = 2.0
        float rgb_to_jpeg_scale_y = static_cast<float>(jpeg_height) / rgb_height; // 720/360 = 2.0

        size_t offset = 0;
        const uint8_t* data_ptr = data.data();

        // Parse detections for each class
        for (size_t class_id = 0; class_id < num_classes; ++class_id) {
            if (offset + sizeof(float32_t) > data.size()) break;

            // Read number of detections for this class
            auto det_count = *reinterpret_cast<const float32_t*>(data_ptr + offset);
            offset += sizeof(float32_t);

            uint32_t count = static_cast<uint32_t>(det_count);
            if (count > 1000) break; // Sanity check to prevent excessive allocations

            // Parse each detection for this class
            for (uint32_t i = 0; i < count; ++i) {
                if (offset + sizeof(hailo_bbox_float32_t) > data.size()) break;

                // Read the bounding box structure
                hailo_bbox_float32_t bbox = *reinterpret_cast<const hailo_bbox_float32_t*>(data_ptr + offset);
                offset += sizeof(hailo_bbox_float32_t);

                // Filter by confidence threshold
                if (bbox.score < confidence_threshold) {
                    continue;
                }

                Detection det;
                det.class_id = static_cast<int>(class_id + 1);  // COCO uses 1-based IDs
                det.confidence = bbox.score;

                // Step 3: Multi-stage coordinate transformation

                // Stage 1: Convert from normalized (0-1) to model input coordinates (640x640)
                float x1_model = bbox.x_min * model_input_size;
                float y1_model = bbox.y_min * model_input_size;
                float x2_model = bbox.x_max * model_input_size;
                float y2_model = bbox.y_max * model_input_size;

                // Stage 2: Remove letterbox padding and scale back to RGB coordinates (640x360)
                // This reverses the letterboxing transformation
                float x1_rgb = (x1_model - pad_x) / scale;
                float y1_rgb = (y1_model - pad_y) / scale;
                float x2_rgb = (x2_model - pad_x) / scale;
                float y2_rgb = (y2_model - pad_y) / scale;

                // Clamp to RGB bounds to handle any precision issues
                x1_rgb = std::max(0.0f, std::min(static_cast<float>(rgb_width), x1_rgb));
                y1_rgb = std::max(0.0f, std::min(static_cast<float>(rgb_height), y1_rgb));
                x2_rgb = std::max(0.0f, std::min(static_cast<float>(rgb_width), x2_rgb));
                y2_rgb = std::max(0.0f, std::min(static_cast<float>(rgb_height), y2_rgb));

                // Stage 3: Scale from RGB coordinates to final JPEG coordinates (1280x720)
                det.x1 = x1_rgb * rgb_to_jpeg_scale_x;
                det.y1 = y1_rgb * rgb_to_jpeg_scale_y;
                det.x2 = x2_rgb * rgb_to_jpeg_scale_x;
                det.y2 = y2_rgb * rgb_to_jpeg_scale_y;

                // Calculate width and height
                det.width = det.x2 - det.x1;
                det.height = det.y2 - det.y1;

                // Final bounds check for JPEG dimensions
                det.x1 = std::max(0.0f, std::min(static_cast<float>(jpeg_width), det.x1));
                det.y1 = std::max(0.0f, std::min(static_cast<float>(jpeg_height), det.y1));
                det.x2 = std::max(0.0f, std::min(static_cast<float>(jpeg_width), det.x2));
                det.y2 = std::max(0.0f, std::min(static_cast<float>(jpeg_height), det.y2));

                // Recalculate dimensions after clamping
                det.width = det.x2 - det.x1;
                det.height = det.y2 - det.y1;

                // Only add detections with valid dimensions
                if (det.width > 0 && det.height > 0) {
                    detections.emplace_back(std::move(det));
                }
            }
        }

        return detections;
    }

    /**
     * @brief JavaScript binding for parsing NMS output
     * 
     * Expected arguments:
     * - buffer: TypedArray containing raw NMS output
     * - numClasses: Number of object classes (e.g., 80 for COCO)
     * - options: Object with parsing options
     *   - threshold: Confidence threshold (default 0.5)
     *   - jpegWidth/Height: Final display dimensions
     *   - rgbWidth/Height: Intermediate RGB dimensions
     *   - modelInputSize: Square model input size
     */
    Napi::Value ParseNMS(const Napi::CallbackInfo &info) {
        Napi::Env env = info.Env();

        // Validate argument count
        if (info.Length() < 3) {
            Napi::TypeError::New(env, "Expected (buffer, numClasses, options)")
                    .ThrowAsJavaScriptException();
            return env.Undefined();
        }

        // Get buffer argument
        if (!info[0].IsTypedArray()) {
            Napi::TypeError::New(env, "First argument must be a typed array")
                    .ThrowAsJavaScriptException();
            return env.Undefined();
        }

        auto typed_array = info[0].As<Napi::TypedArray>();
        auto buffer = typed_array.ArrayBuffer();
        std::span<const uint8_t> data(
            static_cast<uint8_t *>(buffer.Data()) + typed_array.ByteOffset(),
            typed_array.ByteLength()
        );

        // Parse number of classes
        auto num_maybe = info[1].ToNumber();
        if (num_maybe.IsNothing()) {
            Napi::TypeError::New(env, "Second argument must be a number")
                    .ThrowAsJavaScriptException();
            return env.Undefined();
        }
        size_t num_classes = num_maybe.Unwrap().Uint32Value();

        // Parse options object with defaults
        float threshold = 0.5f;
        int jpeg_width = 1280, jpeg_height = 720;    // Final display size
        int rgb_width = 640, rgb_height = 360;       // Intermediate RGB size
        int model_input_size = 640;                  // Square inference size

        if (info[2].IsObject()) {
            auto opts = info[2].As<Napi::Object>();

            // Parse confidence threshold
            bool hasThreshold = false;
            if (opts.Has("threshold").UnwrapTo(&hasThreshold) && hasThreshold) {
                auto val = opts.Get("threshold");
                if (!val.IsNothing() && val.Unwrap().IsNumber()) {
                    threshold = val.Unwrap().As<Napi::Number>().FloatValue();
                }
            }

            // Parse final display dimensions (JPEG)
            bool hasJpegWidth = false;
            if (opts.Has("jpegWidth").UnwrapTo(&hasJpegWidth) && hasJpegWidth) {
                auto val = opts.Get("jpegWidth");
                if (!val.IsNothing() && val.Unwrap().IsNumber()) {
                    jpeg_width = val.Unwrap().As<Napi::Number>().Int32Value();
                }
            }

            bool hasJpegHeight = false;
            if (opts.Has("jpegHeight").UnwrapTo(&hasJpegHeight) && hasJpegHeight) {
                auto val = opts.Get("jpegHeight");
                if (!val.IsNothing() && val.Unwrap().IsNumber()) {
                    jpeg_height = val.Unwrap().As<Napi::Number>().Int32Value();
                }
            }

            // Parse RGB dimensions (intermediate)
            bool hasRgbWidth = false;
            if (opts.Has("rgbWidth").UnwrapTo(&hasRgbWidth) && hasRgbWidth) {
                auto val = opts.Get("rgbWidth");
                if (!val.IsNothing() && val.Unwrap().IsNumber()) {
                    rgb_width = val.Unwrap().As<Napi::Number>().Int32Value();
                }
            }

            bool hasRgbHeight = false;
            if (opts.Has("rgbHeight").UnwrapTo(&hasRgbHeight) && hasRgbHeight) {
                auto val = opts.Get("rgbHeight");
                if (!val.IsNothing() && val.Unwrap().IsNumber()) {
                    rgb_height = val.Unwrap().As<Napi::Number>().Int32Value();
                }
            }

            // Parse model input size (square)
            bool hasModelInputSize = false;
            if (opts.Has("modelInputSize").UnwrapTo(&hasModelInputSize) && hasModelInputSize) {
                auto val = opts.Get("modelInputSize");
                if (!val.IsNothing() && val.Unwrap().IsNumber()) {
                    model_input_size = val.Unwrap().As<Napi::Number>().Int32Value();
                }
            }

            // Legacy support - map old parameter names
            bool hasFrameWidth = false;
            if (opts.Has("frameWidth").UnwrapTo(&hasFrameWidth) && hasFrameWidth) {
                auto val = opts.Get("frameWidth");
                if (!val.IsNothing() && val.Unwrap().IsNumber()) {
                    jpeg_width = val.Unwrap().As<Napi::Number>().Int32Value();
                }
            }

            bool hasFrameHeight = false;
            if (opts.Has("frameHeight").UnwrapTo(&hasFrameHeight) && hasFrameHeight) {
                auto val = opts.Get("frameHeight");
                if (!val.IsNothing() && val.Unwrap().IsNumber()) {
                    jpeg_height = val.Unwrap().As<Napi::Number>().Int32Value();
                }
            }
        }

        // Parse detections with multi-stage coordinate transformation
        auto detections = parse_nms_output(
            data, num_classes, threshold,
            jpeg_width, jpeg_height,
            rgb_width, rgb_height,
            model_input_size
        );

        // Create JavaScript array result
        Napi::Array result = Napi::Array::New(env, detections.size());

        for (size_t i = 0; i < detections.size(); ++i) {
            const auto &det = detections[i];
            Napi::Object obj = Napi::Object::New(env);

            // Populate detection object
            obj["classId"] = det.class_id;
            obj["className"] = std::string(
                det.class_id > 0 && static_cast<size_t>(det.class_id) < COCO_CLASSES.size()
                    ? COCO_CLASSES[det.class_id]
                    : "unknown"
            );
            obj["confidence"] = det.confidence;
            obj["x1"] = det.x1;
            obj["y1"] = det.y1;
            obj["x2"] = det.x2;
            obj["y2"] = det.y2;
            obj["width"] = det.width;
            obj["height"] = det.height;

            result[i] = obj;
        }

        return result;
    }

    /**
     * @brief Get human-readable class name from class ID
     * 
     * Maps COCO dataset class IDs to their string names.
     * Returns "unknown" for invalid IDs.
     */
    Napi::Value GetClassName(const Napi::CallbackInfo &info) {
        if (!info[0].IsNumber()) {
            return Napi::String::New(info.Env(), "unknown");
        }

        auto num_maybe = info[0].ToNumber();
        if (num_maybe.IsNothing()) {
            return Napi::String::New(info.Env(), "unknown");
        }

        int class_id = num_maybe.Unwrap().Int32Value();

        // Check bounds and return appropriate name
        if (class_id >= 0 && static_cast<size_t>(class_id) < COCO_CLASSES.size()) {
            return Napi::String::New(info.Env(), std::string(COCO_CLASSES[class_id]));
        }

        return Napi::String::New(info.Env(), "unknown");
    }
} // namespace hailo_addon
