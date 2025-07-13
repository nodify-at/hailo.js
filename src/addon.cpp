/**
 * @file addon.cpp
 * @brief Main entry point for Hailo Node.js addon
 * @date 2025
 * 
 * This file initializes the Node.js addon and exposes all
 * Hailo functionality to JavaScript.
 */

#include <napi.h>
#include <hailo/hailort.hpp>
#include <hailo_device.hpp>
#include <detection_utils.hpp>

namespace {

/**
 * @brief Scan for available Hailo devices on the system
 * 
 * This function uses the HailoRT API to enumerate all connected
 * Hailo devices. Useful for multi-device systems or debugging.
 * 
 * @param info N-API callback info (no arguments expected)
 * @return Array of device identifier strings
 * 
 * @example
 * ```javascript
 * const devices = hailo.scanDevices();
 * console.log(`Found ${devices.length} Hailo devices`);
 * ```
 */
Napi::Value ScanDevices(const Napi::CallbackInfo& info) {
    auto devices = hailort::Device::scan();
    if (!devices) {
        // Return empty array if scan fails
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

/**
 * @brief Initialize the Node.js addon
 * 
 * This function is called by Node.js when the addon is loaded.
 * It sets up all exported classes and functions.
 * 
 * @param env The N-API environment
 * @param exports The exports object to populate
 * @return The populated exports object
 */
Napi::Object Init(const Napi::Env env, Napi::Object exports) {
    // Initialize and export the HailoDevice class
    hailo_addon::HailoDevice::Init(env, exports);

    // Export standalone utility functions
    exports.Set("scanDevices", Napi::Function::New(env, ScanDevices));

    // Create and export the detection utilities namespace
    const Napi::Object detection = Napi::Object::New(env);
    detection.Set("parseNMS", Napi::Function::New(env, hailo_addon::ParseNMS));
    detection.Set("getClassName", Napi::Function::New(env, hailo_addon::GetClassName));
    (void) exports.Set("detection", detection);

    // Export version information
    exports.Set("version", Napi::String::New(env, "2.0.0"));

    return exports;
}

// Register the addon with Node.js
NODE_API_MODULE(hailo, Init)
