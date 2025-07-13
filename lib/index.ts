/**
 * Hailo.js - TypeScript bindings for Hailo AI inference
 *
 * @packageDocumentation
 */

export { HailoDevice } from './hailo-device.js'
export * from './detection.js'
export * from './types.js'
export * from './utils.js'

// Re-export commonly used types at top level
export type {
    Detection,
    InferenceInputs,
    InferenceOutputs,
    ModelInfo,
    NMSOptions,
    BoundingBox,
    ModelConfig,
    PerformanceMetrics,
} from './types.js'

// Export model type enum for convenience
export { ModelType } from './types.js'
