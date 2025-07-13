#include <napi.h>
#include <hailo/hailort.hpp>
#include <hailo_device.hpp>
#include <detection_utils.hpp>

namespace {

Napi::Value ScanDevices(const Napi::CallbackInfo& info) {
    auto devices = hailort::Device::scan();
    if (!devices) {
        return Napi::Array::New(info.Env(), 0);
    }

    auto& device_list = devices.value();
    Napi::Array result = Napi::Array::New(info.Env(), device_list.size());

    for (size_t i = 0; i < device_list.size(); ++i) {
        result[i] = Napi::String::New(info.Env(), device_list[i]);
    }

    return result;
}

} // anonymous namespace

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // Device class
    hailo_addon::HailoDevice::Init(env, exports);

    // Standalone functions
    exports.Set("scanDevices", Napi::Function::New(env, ScanDevices));

    // Detection utilities
    Napi::Object detection = Napi::Object::New(env);
    detection.Set("parseNMS", Napi::Function::New(env, hailo_addon::ParseNMS));
    detection.Set("getClassName", Napi::Function::New(env, hailo_addon::GetClassName));
    exports.Set("detection", detection);

    // Version
    exports.Set("version", Napi::String::New(env, "2.0.0"));

    return exports;
}

NODE_API_MODULE(hailo, Init)
