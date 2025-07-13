/**
 * Type definitions for the native Hailo addon
 * These types describe the C++ addon interface
 */

/**
 * Input data for inference - maps input names to typed arrays
 */
export interface NativeInferenceInputs {
    [inputName: string]: Uint8Array | Float32Array
}

/**
 * Output data from inference - maps output names to typed arrays
 */
export interface NativeInferenceOutputs {
    [outputName: string]: Uint8Array | Float32Array
}

/**
 * Model information including input/output tensor names
 */
export interface NativeModelInfo {
    inputs: string[]
    outputs: string[]
    active: boolean
    inferenceCount: number
}

/**
 * Options for NMS (Non-Maximum Suppression) parsing
 */
export interface NativeNMSOptions {
    threshold?: number
    frameWidth?: number
    frameHeight?: number
}

/**
 * Detection result from object detection models
 */
export interface NativeDetection {
    classId: number
    className: string
    confidence: number
    x1: number
    y1: number
    x2: number
    y2: number
    width: number
    height: number
}

/**
 * Native Hailo device class interface
 */
export interface NativeHailoDeviceClass {
    new (): NativeHailoDevice
}

/**
 * Native Hailo device instance interface
 */
export interface NativeHailoDevice {
    /**
     * Load a compiled Hailo model (HEF file)
     * @param modelPath Path to the .hef file
     * @returns Promise that resolves when model is loaded
     */
    loadModel(modelPath: string): Promise<boolean>

    /**
     * Run inference on the loaded model
     * @param inputs Input tensors mapped by name
     * @returns Promise with output tensors
     */
    infer(inputs: NativeInferenceInputs): Promise<NativeInferenceOutputs>

    /**
     * Get information about the loaded model
     * @returns Model metadata including tensor names
     */
    getModelInfo(): NativeModelInfo

    /**
     * Check if a model is loaded and ready
     * @returns True if model is active
     */
    isActive(): boolean
}

/**
 * Native detection utilities interface
 */
export interface NativeDetectionUtils {
    /**
     * Parse NMS output from YOLO-style models
     * @param buffer Raw output buffer from model
     * @param numClasses Number of object classes
     * @param options Parsing options
     * @returns Array of detected objects
     */
    parseNMS(buffer: Uint8Array | Float32Array, numClasses: number, options?: NativeNMSOptions): NativeDetection[]

    /**
     * Get human-readable class name from ID
     * @param classId Numeric class ID
     * @returns Class name string
     */
    getClassName(classId: number): string
}

/**
 * Complete native addon interface
 */
export interface NativeAddon {
    /**
     * Hailo device class constructor
     */
    HailoDevice: NativeHailoDeviceClass

    /**
     * Detection utilities
     */
    detection: NativeDetectionUtils

    /**
     * Scan for available Hailo devices
     * @returns Array of device identifiers
     */
    scanDevices(): string[]

    /**
     * Addon version
     */
    version: string
}
