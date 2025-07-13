/**
 * Public TypeScript type definitions for Hailo.js
 * Re-exports and extends native addon types with additional utilities
 */

// Type declaration for node-gyp-build
export type NodeGypBuild = (dir: string) => unknown

export interface InferenceInputs {
    [inputName: string]: Uint8Array | Float32Array
}

export interface InferenceOutputs {
    [outputName: string]: Uint8Array | Float32Array
}

export interface ModelInfo {
    inputs: string[]
    outputs: string[]
    active: boolean
    inferenceCount: number
}

export interface Detection {
    classId: number
    className: string
    confidence: number
    x1: number
    y1: number
    x2: number
    y2: number
    width: number
    height: number
    trackingId?: string
    visibleFrames?: number
    hiddenFrames?: number
}

export interface BoundingBox {
    x: number
    y: number
    width: number
    height: number
}

export interface NMSOptions {
    /**
     * Confidence threshold for detections (0-1)
     * @default 0.5
     */
    threshold?: number

    jpegWidth?: number // Final display size
    jpegHeight?: number
    rgbWidth?: number // Intermediate RGB size
    rgbHeight?: number
    modelInputSize?: number // Square inference size
}

export interface DeviceOptions {
    /**
     * Enable debug logging
     * @default false
     */
    debug?: boolean

    /**
     * Device index to use (if multiple devices)
     * @default 0
     */
    deviceIndex?: number
}

/**
 * Supported model types
 */
export enum ModelType {
    YOLO = 'yolo',
    YOLOX = 'yolox',
    YOLOV8 = 'yolov8',
    CUSTOM = 'custom',
}

/**
 * Model configuration
 */
export interface ModelConfig {
    type: ModelType
    path: string
    inputName?: string
    outputName?: string
    numClasses?: number
}

/**
 * Performance metrics
 */
export interface PerformanceMetrics {
    inferenceTime: number
    preprocessTime?: number
    postprocessTime?: number
    totalTime: number
    fps: number
}
