#include <hailo_device.hpp>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>

namespace hailo_addon {
    Napi::FunctionReference HailoDevice::constructor;

    Napi::Object HailoDevice::Init(Napi::Env env, Napi::Object exports) {
        Napi::Function func = DefineClass(env, "HailoDevice", {
                                              InstanceMethod<&HailoDevice::LoadModel>("loadModel"),
                                              InstanceMethod<&HailoDevice::Infer>("infer"),
                                              InstanceMethod<&HailoDevice::GetModelInfo>("getModelInfo"),
                                              InstanceMethod<&HailoDevice::IsActive>("isActive")
                                          });

        constructor = Napi::Persistent(func);
        constructor.SuppressDestruct();

        exports.Set("HailoDevice", func);
        return exports;
    }

    HailoDevice::HailoDevice(const Napi::CallbackInfo &info)
        : Napi::ObjectWrap<HailoDevice>(info) {
        Napi::Env env = info.Env();

        // Create VDevice with default params
        hailo_vdevice_params_t params{};
        hailo_init_vdevice_params(&params);
        params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;

        auto vdevice_exp = hailort::VDevice::create(params);
        if (!vdevice_exp) {
            std::stringstream ss;
            ss << "Failed to create VDevice: " << vdevice_exp.status();
            Napi::Error::New(env, ss.str()).ThrowAsJavaScriptException();
            return;
        }

        vdevice_ = vdevice_exp.release();
    }

    HailoDevice::~HailoDevice() {
        is_active_ = false;
    }

    Napi::Value HailoDevice::LoadModel(const Napi::CallbackInfo &info) {
        Napi::Env env = info.Env();

        if (!info[0].IsString()) {
            Napi::TypeError::New(env, "Model path must be a string").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        std::string hef_path = info[0].As<Napi::String>().Utf8Value();

        // Create promise
        auto deferred = Napi::Promise::Deferred::New(env);

        // Create ThreadSafeFunction for calling back to JS
        auto tsfn = Napi::ThreadSafeFunction::New(
            env,
            Napi::Function::New(env, [](const Napi::CallbackInfo &) {
            }), // dummy function
            "LoadModelCallback",
            0,
            1
        );

        // Lambda to perform the load operation
        auto load_operation = [this, hef_path, deferred, tsfn]() {
            struct Result {
                bool success;
                std::string error;
            } result{true, ""};

            try {
                std::lock_guard lock(infer_mutex_);

                // Create infer model
                auto model_exp = vdevice_->create_infer_model(hef_path);
                if (!model_exp) {
                    std::stringstream ss;
                    ss << "Failed to create model: " << model_exp.status();
                    result.success = false;
                    result.error = ss.str();
                } else {
                    model_ = model_exp.release();

                    // Configure model
                    auto configured_exp = model_->configure();
                    if (!configured_exp) {
                        std::stringstream ss;
                        ss << "Failed to configure model: " << configured_exp.status();
                        result.success = false;
                        result.error = ss.str();
                    } else {
                        configured_model_ = std::make_shared<hailort::ConfiguredInferModel>(configured_exp.release());

                        // Create bindings
                        auto bindings_exp = configured_model_->create_bindings();
                        if (!bindings_exp) {
                            std::stringstream ss;
                            ss << "Failed to create bindings: " << bindings_exp.status();
                            result.success = false;
                            result.error = ss.str();
                        } else {
                            bindings_ = std::make_unique<hailort::ConfiguredInferModel::Bindings>(
                                bindings_exp.release());

                            // Cache model info
                            input_names_ = model_->get_input_names();
                            output_names_ = model_->get_output_names();

                            output_sizes_.clear();
                            for (const auto &name: output_names_) {
                                auto output = model_->output(name);
                                if (output) {
                                    output_sizes_[name] = output->get_frame_size();
                                }
                            }

                            is_active_ = true;
                        }
                    }
                }
            } catch (const std::exception &e) {
                result.success = false;
                result.error = e.what();
            }

            // Call back to JS thread
            auto callback = [deferred, result](Napi::Env env, Napi::Function) {
                Napi::HandleScope scope(env);
                if (result.success) {
                    deferred.Resolve(Napi::Boolean::New(env, true));
                } else {
                    deferred.Reject(Napi::Error::New(env, result.error).Value());
                }
            };

            tsfn.BlockingCall(callback);
            tsfn.Release();
        };

        // Execute async
        std::thread(load_operation).detach();

        return deferred.Promise();
    }

    Napi::Value HailoDevice::Infer(const Napi::CallbackInfo &info) {
        Napi::Env env = info.Env();

        if (!info[0].IsObject()) {
            Napi::TypeError::New(env, "Inputs must be an object").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        if (!is_active_) {
            Napi::Error::New(env, "Model not loaded").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        Napi::Object inputs_obj = info[0].As<Napi::Object>();
        auto deferred = Napi::Promise::Deferred::New(env);

        // Prepare input map with proper lifetime management
        struct InferContext {
            std::unordered_map<std::string, std::vector<uint8_t> > owned_data;
            std::unordered_map<std::string, std::span<const uint8_t> > input_spans;
            uint64_t infer_id;
        };

        auto context = std::make_shared<InferContext>();
        context->infer_id = ++infer_count_;

        // Extract and copy input data
        auto prop_names_maybe = inputs_obj.GetPropertyNames();
        if (prop_names_maybe.IsNothing()) {
            Napi::Error::New(env, "Failed to get property names").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        Napi::Array prop_names = prop_names_maybe.Unwrap();
        for (uint32_t i = 0; i < prop_names.Length(); ++i) {
            auto prop_maybe = prop_names.Get(i);
            if (prop_maybe.IsNothing()) continue;

            auto prop = prop_maybe.Unwrap();
            auto name_maybe = prop.ToString();
            if (name_maybe.IsNothing()) continue;
            std::string name = name_maybe.Unwrap().Utf8Value();

            auto value_maybe = inputs_obj.Get(name);
            if (value_maybe.IsNothing()) continue;

            auto value = value_maybe.Unwrap();
            if (value.IsTypedArray()) {
                auto typed_array = value.As<Napi::TypedArray>();
                auto buffer = typed_array.ArrayBuffer();

                size_t byte_offset = typed_array.ByteOffset();
                size_t byte_length = typed_array.ByteLength();

                // Copy data to ensure it stays valid
                context->owned_data[name].resize(byte_length);
                std::memcpy(context->owned_data[name].data(),
                            static_cast<uint8_t *>(buffer.Data()) + byte_offset,
                            byte_length);

                context->input_spans[name] = std::span<const uint8_t>(
                    context->owned_data[name].data(),
                    context->owned_data[name].size()
                );
            }
        }

        // Create ThreadSafeFunction for calling back to JS
        auto tsfn = Napi::ThreadSafeFunction::New(
            env,
            Napi::Function::New(env, [](const Napi::CallbackInfo &) {
            }), // dummy function
            "InferCallback",
            0,
            1
        );

        // Perform inference async
        auto infer_operation = [this, context, deferred, tsfn]() {
            // Use shared_ptr to avoid copy issues with unique_ptr
            struct OutputBuffer {
                uint8_t *data;
                size_t size;
                hailo_format_type_t format;

                OutputBuffer(size_t sz) : size(sz), format(HAILO_FORMAT_TYPE_UINT8) {
                    constexpr size_t alignment = 64;
                    size_t aligned_size = ((sz + alignment - 1) / alignment) * alignment;
                    data = new uint8_t[aligned_size];
                }

                ~OutputBuffer() {
                    delete[] data;
                }

                // Delete copy constructor and assignment
                OutputBuffer(const OutputBuffer &) = delete;

                OutputBuffer & operator=(const OutputBuffer &) = delete;

                // Allow move
                OutputBuffer(OutputBuffer &&other) noexcept
                    : data(other.data), size(other.size), format(other.format) {
                    other.data = nullptr;
                }
            };

            struct Result {
                bool success = true;
                std::string error;
                std::vector<std::pair<std::string, std::shared_ptr<OutputBuffer> > > outputs;
            };

            auto result = std::make_shared<Result>();

            try {
                std::lock_guard lock(infer_mutex_);

                if (!configured_model_ || !bindings_) {
                    result->success = false;
                    result->error = "Model not properly initialized";
                } else {
                    // Set input buffers
                    for (const auto &[name, data]: context->input_spans) {
                        auto input_binding = bindings_->input(name);
                        if (!input_binding) {
                            std::stringstream ss;
                            ss << "Unknown input: " << name;
                            result->success = false;
                            result->error = ss.str();
                            break;
                        }

                        auto status = input_binding->set_buffer(hailort::MemoryView(
                            const_cast<uint8_t *>(data.data()), data.size()
                        ));

                        if (status != HAILO_SUCCESS) {
                            std::stringstream ss;
                            ss << "Failed to set input buffer for " << name << ": " << status;
                            result->success = false;
                            result->error = ss.str();
                            break;
                        }
                    }

                    if (result->success) {
                        // Allocate output buffers
                        std::unordered_map<std::string, std::shared_ptr<OutputBuffer> > temp_buffers;

                        for (const auto &[name, size]: output_sizes_) {
                            auto buffer = std::make_shared<OutputBuffer>(size);

                            auto output_binding = bindings_->output(name);
                            if (!output_binding) {
                                std::stringstream ss;
                                ss << "Unknown output: " << name;
                                result->success = false;
                                result->error = ss.str();
                                break;
                            }

                            auto status = output_binding->set_buffer(hailort::MemoryView(
                                buffer->data, size
                            ));

                            if (status != HAILO_SUCCESS) {
                                std::stringstream ss;
                                ss << "Failed to set output buffer for " << name << ": " << status;
                                result->success = false;
                                result->error = ss.str();
                                break;
                            }

                            // Store format info
                            auto output = model_->output(name);
                            if (output) {
                                buffer->format = output->format().type;
                            }

                            temp_buffers[name] = buffer;
                        }

                        if (result->success) {
                            // Run inference
                            auto job = configured_model_->run_async(*bindings_);
                            if (!job) {
                                std::stringstream ss;
                                ss << "Failed to start inference: " << job.status();
                                result->success = false;
                                result->error = ss.str();
                            } else {
                                // Wait for completion
                                auto wait_status = job->wait(std::chrono::milliseconds(1000));
                                if (wait_status != HAILO_SUCCESS) {
                                    std::stringstream ss;
                                    ss << "Inference failed: " << wait_status;
                                    result->success = false;
                                    result->error = ss.str();
                                } else {
                                    // Move buffers to result
                                    for (auto &[name, buffer]: temp_buffers) {
                                        result->outputs.emplace_back(name, buffer);
                                    }
                                }
                            }
                        }
                    }
                }
            } catch (const std::exception &e) {
                result->success = false;
                result->error = e.what();
            }

            // Call back to JS thread
            auto callback = [deferred, result](Napi::Env env, Napi::Function) {
                Napi::HandleScope scope(env);

                if (!result->success) {
                    deferred.Reject(Napi::Error::New(env, result->error).Value());
                    return;
                }

                // Create result object with zero-copy buffers
                Napi::Object js_result = Napi::Object::New(env);

                for (const auto &[name, buffer]: result->outputs) {
                    // Transfer ownership to JS
                    uint8_t *raw_data = buffer->data;
                    size_t size = buffer->size;
                    buffer->data = nullptr; // Prevent deletion

                    // Create external buffer that owns the memory
                    auto array_buffer = Napi::ArrayBuffer::New(
                        env,
                        raw_data,
                        size,
                        [](Napi::Env, void *data) {
                            delete[] static_cast<uint8_t *>(data);
                        }
                    );

                    if (buffer->format == HAILO_FORMAT_TYPE_FLOAT32) {
                        js_result.Set(name, Napi::Float32Array::New(env, size / sizeof(float), array_buffer, 0));
                    } else {
                        js_result.Set(name, Napi::Uint8Array::New(env, size, array_buffer, 0));
                    }
                }

                deferred.Resolve(js_result);
            };

            tsfn.BlockingCall(callback);
            tsfn.Release();
        };

        std::thread(infer_operation).detach();

        return deferred.Promise();
    }

    Napi::Value HailoDevice::GetModelInfo(const Napi::CallbackInfo &info) {
        Napi::Env env = info.Env();
        Napi::Object result = Napi::Object::New(env);

        auto to_js_array = [&env](const std::vector<std::string> &vec) {
            Napi::Array arr = Napi::Array::New(env, vec.size());
            for (size_t i = 0; i < vec.size(); ++i) {
                arr[i] = Napi::String::New(env, vec[i]);
            }
            return arr;
        };

        result.Set("inputs", to_js_array(input_names_));
        result.Set("outputs", to_js_array(output_names_));
        result.Set("active", Napi::Boolean::New(env, is_active_.load()));
        result.Set("inferenceCount", Napi::Number::New(env, infer_count_.load()));

        return result;
    }

    Napi::Value HailoDevice::IsActive(const Napi::CallbackInfo &info) {
        return Napi::Boolean::New(info.Env(), is_active_.load());
    }
} // namespace hailo_addon
