import { createRequire } from 'node:module'

/**
 * HailoDevice - High-level wrapper for Hailo AI inference
 * Provides a clean, type-safe interface with performance monitoring
 */
import { EventEmitter } from 'events'
import type { NativeAddon, NativeHailoDevice } from './native-types.js'
import {
    InferenceInputs,
    InferenceOutputs,
    ModelInfo,
    DeviceOptions,
    ModelConfig,
    PerformanceMetrics,
    ModelType,
    NodeGypBuild,
} from './types.js'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { join } from 'path'

// Create require for CommonJS modules
const require = createRequire(import.meta.url)
const __dirname = dirname(fileURLToPath(import.meta.url))

// Load node-gyp-build and cast it
const nodeGypBuild = require('node-gyp-build') as NodeGypBuild
const addon = nodeGypBuild(join(__dirname, '..')) as NativeAddon

/**
 * High-level Hailo device interface with enhanced functionality
 *
 * @example
 * ```typescript
 * const device = new HailoDevice();
 * await device.loadModel({
 *   type: ModelType.YOLOV8,
 *   path: './models/yolov8n.hef',
 *   numClasses: 80
 * });
 *
 * const output = await device.infer(inputTensor);
 * ```
 */
export class HailoDevice extends EventEmitter {
    private device: NativeHailoDevice
    private modelConfig?: ModelConfig
    private performanceBuffer: PerformanceMetrics[] = []
    private readonly options: Required<DeviceOptions>

    constructor(options: DeviceOptions = {}) {
        super()

        this.options = { debug: false, deviceIndex: 0, ...options }

        // Create native device instance
        this.device = new addon.HailoDevice()

        if (this.options.debug) {
            this.on('debug', msg => console.log(`[HailoDevice] ${msg}`))
        }
    }

    /**
     * Load a model with configuration
     * Supports automatic model type detection from filename
     */
    async loadModel(config: ModelConfig | string): Promise<void> {
        const startTime = performance.now()

        // Handle string path with auto-detection
        if (typeof config === 'string') {
            config = this.detectModelConfig(config)
        }

        this.modelConfig = config
        this.emit('debug', `Loading model: ${config.path} (type: ${config.type})`)

        try {
            const success = await this.device.loadModel(config.path)
            if (!success) {
                throw new Error('Failed to load model')
            }

            const loadTime = performance.now() - startTime
            this.emit('modelLoaded', { config, loadTime, info: this.device.getModelInfo() })

            this.emit('debug', `Model loaded in ${loadTime.toFixed(2)}ms`)
        } catch (error) {
            this.emit('error', error)
            throw error
        }
    }

    /**
     * Run inference with performance tracking
     */
    async infer(inputs: InferenceInputs): Promise<InferenceOutputs> {
        if (!this.device.isActive()) {
            throw new Error('No model loaded')
        }

        const metrics: PerformanceMetrics = { inferenceTime: 0, totalTime: 0, fps: 0 }

        const startTime = performance.now()

        try {
            // Run inference
            const inferStart = performance.now()
            const outputs = await this.device.infer(inputs)
            metrics.inferenceTime = performance.now() - inferStart

            metrics.totalTime = performance.now() - startTime
            metrics.fps = 1000 / metrics.totalTime

            // Track performance
            this.trackPerformance(metrics)
            this.emit('inference', { outputs, metrics })

            return outputs
        } catch (error) {
            this.emit('error', error)
            throw error
        }
    }

    /**
     * Get model information
     */
    getModelInfo(): ModelInfo {
        return this.device.getModelInfo()
    }

    /**
     * Check if device is ready for inference
     */
    isReady(): boolean {
        return this.device.isActive()
    }

    /**
     * Get average performance metrics
     */
    getPerformanceStats(): {
        avgInferenceTime: number
        avgFps: number
        minInferenceTime: number
        maxInferenceTime: number
    } {
        if (this.performanceBuffer.length === 0) {
            return { avgInferenceTime: 0, avgFps: 0, minInferenceTime: 0, maxInferenceTime: 0 }
        }

        const times = this.performanceBuffer.map(m => m.inferenceTime)
        const sum = times.reduce((a, b) => a + b, 0)

        return {
            avgInferenceTime: sum / times.length,
            avgFps: this.performanceBuffer.reduce((a, b) => a + b.fps, 0) / this.performanceBuffer.length,
            minInferenceTime: Math.min(...times),
            maxInferenceTime: Math.max(...times),
        }
    }

    /**
     * Reset performance tracking
     */
    resetPerformanceStats(): void {
        this.performanceBuffer = []
    }

    /**
     * Get current model configuration
     */
    getModelConfig(): ModelConfig | undefined {
        return this.modelConfig
    }

    /**
     * Detect model configuration from filename
     */
    private detectModelConfig(path: string): ModelConfig {
        const filename = path.toLowerCase()

        let type: ModelType = ModelType.CUSTOM
        let numClasses = 80 // Default COCO classes

        if (filename.includes('yolov8') || filename.includes('yolo8')) {
            type = ModelType.YOLOV8
        } else if (filename.includes('yolox')) {
            type = ModelType.YOLOX
        } else if (filename.includes('yolo')) {
            type = ModelType.YOLO
        }

        // Try to detect number of classes from filename
        const classMatch = filename.match(/(\d+)classes?/)
        if (classMatch) {
            numClasses = parseInt(classMatch[1]!, 10)
        }

        return { type, path, numClasses }
    }

    /**
     * Track performance metrics with sliding window
     */
    private trackPerformance(metrics: PerformanceMetrics): void {
        this.performanceBuffer.push(metrics)

        // Keep last 100 measurements
        if (this.performanceBuffer.length > 100) {
            this.performanceBuffer.shift()
        }
    }

    /**
     * Static method to scan for available devices
     */
    static async scanDevices(): Promise<string[]> {
        return addon.scanDevices()
    }

    /**
     * Get addon version
     */
    static get version(): string {
        return addon.version
    }
}

// Re-export types
export * from './types.js'
