# Hailo.js 🚀

> ⚠️ **Experimental**: This project is in active development and APIs may change.

High-performance Node.js bindings for Hailo AI inference accelerators with real-time object detection and streaming capabilities.

<p align="center">
  <img src="https://img.shields.io/npm/v/@hailo/js.svg" alt="npm version" />
  <img src="https://img.shields.io/badge/node-%3E%3D18-green.svg" alt="node version" />
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license" />
</p>

## 🌟 Features

- **Native Performance**: Direct C++ bindings to Hailo RT for maximum inference speed
- **TypeScript First**: Full TypeScript support with comprehensive type definitions
- **Real-time Streaming**: WebSocket-based live detection server with browser visualization
- **Temporal Filtering**: Advanced object tracking with velocity prediction and smoothing
- **Multi-Model Support**: Built-in support for YOLO, YOLOv8, YOLOX with automatic model detection
- **Production Ready**: Performance monitoring, graceful shutdown, and error handling
- **Easy Integration**: Simple API for both batch processing and streaming applications

## 📋 Requirements

- Node.js >= 20
- Hailo hardware (Hailo-8, Hailo-8L, etc.) and Raspberry Pi 5 (tested on Raspberry Pi OS with PI 5)
- HailoRT runtime installed
- Raspberry OS

## 🚀 Quick Start

### Installation

```bash
npm install @nodify_at/hailo.js
```

### Basic Usage

```typescript
import { HailoDevice, parseNMS, ModelType } from '@hailo/js';

// Initialize device
const device = new HailoDevice({ debug: true });

// Load model
await device.loadModel({
  type: ModelType.YOLOV8,
  path: './models/yolov8n.hef',
  numClasses: 80
});

// Prepare input (RGB image as Uint8Array)
const input = new Uint8Array(640 * 640 * 3);

// Run inference
const outputs = await device.infer({
  'yolov8n/input_layer1': input
});

// Parse detections
const detections = parseNMS(
  outputs['yolov8n/yolov8_nms_postprocess'],
  80,
  { threshold: 0.5 }
);

console.log(`Found ${detections.length} objects`);
```

## 🎥 Live Detection Server

Start a real-time object detection server with WebSocket streaming:

```typescript
import { LiveDetectionServer } from '@hailo/js/server';

const server = new LiveDetectionServer({
  port: 8080,
  modelPath: 'yolov8n.hef'
});

await server.start();
// Server available at ws://localhost:8080
```

Open `index.html` in your browser to see live detections with:
- Real-time video streaming
- Smooth object tracking with temporal filtering
- Performance metrics
- Debug visualization

## 📚 API Documentation

### HailoDevice

Main device interface for inference:

```typescript
const device = new HailoDevice(options?: DeviceOptions);

// Load a model
await device.loadModel(config: ModelConfig | string);

// Run inference
const outputs = await device.infer(inputs: InferenceInputs);

// Get model info
const info = device.getModelInfo();

// Check if ready
const ready = device.isReady();

// Get performance stats
const stats = device.getPerformanceStats();
```

### Detection Utilities

```typescript
// Parse NMS output
const detections = parseNMS(buffer, numClasses, options);

// Filter detections
const people = filterByClasses(detections, ['person']);
const highConf = filterByConfidence(detections, 0.7);

// Calculate IoU
const iou = calculateIoU(det1, det2);

// Group by class
const groups = groupByClass(detections);
```

### Temporal Filter

Advanced tracking with velocity prediction:

```typescript
const filter = new TemporalFilter({
  iouThreshold: 0.1,
  minVisibleFrames: 5,
  useVelocityPrediction: true
});

// Process frame
const filtered = filter.processFrame(detections);
```

## 🏗️ Architecture (for demo)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Camera Input  │────▶│ Detection Server │────▶│ WebSocket Clients│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │ Hailo Device │
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │ Temporal Filter  │
                        └──────────────────┘
```

## ⚡ Performance

- **Inference Speed**: Up to 30 FPS on Hailo-8 with YOLOv8
- **Low Latency**: < 50ms end-to-end processing time
- **Efficient Streaming**: Hardware-accelerated JPEG encoding
- **Optimized Tracking**: Minimal overhead temporal filtering

## 🛠️ Advanced Configuration

### Custom Detection Server

```typescript
const server = new LiveDetectionServer({
  port: 8080,
  modelPath: 'custom-model.hef',
  camera: {
    width: 1920,
    height: 1080,
    fps: 30
  },
  detection: {
    confidenceThreshold: 0.45,
    processEveryNFrames: 2
  },
  temporal: {
    minVisibleFrames: 5,
    useVelocityPrediction: true,
    classSpecificThresholds: new Map([
      [1, { iou: 0.15, confidence: 0.6 }], // person
      [3, { iou: 0.1, confidence: 0.55 }]  // car
    ])
  }
});
```

### Event Handling

```typescript
device.on('modelLoaded', ({ config, loadTime }) => {
  console.log(`Model loaded in ${loadTime}ms`);
});

device.on('inference', ({ outputs, metrics }) => {
  console.log(`Inference took ${metrics.inferenceTime}ms`);
});

device.on('error', (error) => {
  console.error('Device error:', error);
});
```

## 📊 Monitoring & Stats

Built-in performance monitoring:

```typescript
const stats = device.getPerformanceStats();
console.log(`Average FPS: ${stats.avgFps}`);
console.log(`Average inference: ${stats.avgInferenceTime}ms`);
```

## 🧪 Examples

Check out the `/examples` directory for:
- Basic detection script
- Live streaming server
- Multi-model comparison
- Custom post-processing

## 🤝 Contributing

Contributions are welcome!

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Hailo team for creating the powerful AI accelerator hardware
- YOLO authors for the excellent detection models
- Node.js N-API team for making native addons possible
- The open source community for inspiration and support

---

<div align="center">
  Made with ❤️ by nodify
</div>