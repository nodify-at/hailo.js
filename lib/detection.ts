/**
 * Detection utilities for object detection models
 * Provides high-level functions for parsing and filtering detections
 */

import type { NativeAddon } from './native-types.js'
import type { Detection, NMSOptions, BoundingBox, NodeGypBuild } from './types.js'
import { createRequire } from 'node:module'
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
 * Parse NMS output from YOLO-style models
 *
 * @example
 * ```typescript
 * const detections = parseNMS(outputBuffer, 80, {
 *   threshold: 0.5,
 *   frameWidth: 1920,
 *   frameHeight: 1080
 * });
 * ```
 */
export function parseNMS(buffer: Uint8Array | Float32Array, numClasses: number, options: NMSOptions = {}): Detection[] {
    const defaultOptions: NMSOptions = { threshold: 0.5, jpegWidth: 1920, jpegHeight: 1080, ...options }
    return addon.detection.parseNMS(buffer, numClasses, defaultOptions)
}

/**
 * Get human-readable class name from ID
 */
export function getClassName(classId: number): string {
    return addon.detection.getClassName(classId)
}

/**
 * Filter detections by class names
 *
 * @example
 * ```typescript
 * const people = filterByClasses(detections, ['person']);
 * const vehicles = filterByClasses(detections, ['car', 'truck', 'bus']);
 * ```
 */
export function filterByClasses(detections: Detection[], classNames: string[]): Detection[] {
    const classNamesSet = new Set(classNames.map(name => name.toLowerCase()))
    return detections.filter(det => classNamesSet.has(det.className.toLowerCase()))
}

/**
 * Filter detections by confidence threshold
 */
export function filterByConfidence(detections: Detection[], threshold: number): Detection[] {
    return detections.filter(det => det.confidence >= threshold)
}

/**
 * Filter detections by area (in pixels)
 */
export function filterByArea(detections: Detection[], minArea: number, maxArea?: number): Detection[] {
    return detections.filter(det => {
        const area = det.width * det.height
        return area >= minArea && (!maxArea || area <= maxArea)
    })
}

/**
 * Convert detection to normalized bounding box (0-1 range)
 */
export function toNormalizedBox(detection: Detection, frameWidth: number, frameHeight: number): BoundingBox {
    return {
        x: detection.x1 / frameWidth,
        y: detection.y1 / frameHeight,
        width: detection.width / frameWidth,
        height: detection.height / frameHeight,
    }
}

/**
 * Calculate IoU (Intersection over Union) between two detections
 */
export function calculateIoU(det1: Detection, det2: Detection): number {
    const x1 = Math.max(det1.x1, det2.x1)
    const y1 = Math.max(det1.y1, det2.y1)
    const x2 = Math.min(det1.x2, det2.x2)
    const y2 = Math.min(det1.y2, det2.y2)

    if (x2 <= x1 || y2 <= y1) {
        return 0
    }

    const intersection = (x2 - x1) * (y2 - y1)
    const area1 = det1.width * det1.height
    const area2 = det2.width * det2.height
    const union = area1 + area2 - intersection

    return intersection / union
}

/**
 * Group detections by class
 */
export function groupByClass(detections: Detection[]): Map<string, Detection[]> {
    const groups = new Map<string, Detection[]>()

    for (const detection of detections) {
        const group = groups.get(detection.className) ?? []
        group.push(detection)
        groups.set(detection.className, group)
    }

    return groups
}

/**
 * Find detections within a specific region
 */
export function findInRegion(
    detections: Detection[],
    region: BoundingBox,
    frameWidth: number,
    frameHeight: number,
    overlap: number = 0.5,
): Detection[] {
    const regionPixels = {
        x1: region.x * frameWidth,
        y1: region.y * frameHeight,
        x2: (region.x + region.width) * frameWidth,
        y2: (region.y + region.height) * frameHeight,
    }

    return detections.filter(det => {
        const intersectX = Math.max(0, Math.min(det.x2, regionPixels.x2) - Math.max(det.x1, regionPixels.x1))
        const intersectY = Math.max(0, Math.min(det.y2, regionPixels.y2) - Math.max(det.y1, regionPixels.y1))

        const intersectArea = intersectX * intersectY
        const detArea = det.width * det.height

        return intersectArea / detArea >= overlap
    })
}

/**
 * Common COCO class names for reference
 */
export const COCO_CLASSES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
] as const

export type COCOClassName = (typeof COCO_CLASSES)[number]
