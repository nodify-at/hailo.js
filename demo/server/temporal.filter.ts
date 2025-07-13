import { Detection } from '../../lib/index.js'

interface BoundingBox {
    x1: number
    y1: number
    x2: number
    y2: number
}

export interface TrackedObject {
    id: string
    classId: number
    className: string
    detection: Detection

    // Visibility tracking
    visibleCount: number
    hiddenCount: number
    isVisible: boolean
    lastSeen: number
    firstSeen: number

    // Motion tracking
    smoothedBox: BoundingBox
    velocity: { vx: number; vy: number }
    predictedBox?: BoundingBox

    // Quality metrics
    avgConfidence: number
    maxConfidence: number
    confidenceHistory: number[]

    // Tracking metadata
    trackingId?: string
    visibleFrames?: number
    hiddenFrames?: number
}

interface FilterConfig {
    // Matching thresholds
    iouThreshold: number
    velocityWeight: number
    confidenceWeight: number

    // Smoothing parameters
    boxSmoothingFactor: number
    velocitySmoothingFactor: number

    // Visibility thresholds
    minVisibleFrames: number
    maxHiddenFrames: number

    // Advanced options
    useVelocityPrediction: boolean
    adaptiveThresholds: boolean
    maxConfidenceHistory: number
    classSpecificThresholds?: Map<number, { iou: number; confidence: number }>
}

export class TemporalFilter {
    private trackedObjects = new Map<string, TrackedObject>()
    private nextObjectId = 1
    private frameNumber = 0
    private config: FilterConfig

    constructor(config: Partial<FilterConfig> = {}) {
        this.config = {
            iouThreshold: config.iouThreshold ?? 0.1,
            velocityWeight: config.velocityWeight ?? 0.2,
            confidenceWeight: config.confidenceWeight ?? 0.1,
            boxSmoothingFactor: config.boxSmoothingFactor ?? 0.75,
            velocitySmoothingFactor: config.velocitySmoothingFactor ?? 0.8,
            minVisibleFrames: config.minVisibleFrames ?? 6,
            maxHiddenFrames: config.maxHiddenFrames ?? 3,
            useVelocityPrediction: config.useVelocityPrediction ?? true,
            adaptiveThresholds: config.adaptiveThresholds ?? true,
            maxConfidenceHistory: config.maxConfidenceHistory ?? 10,
            classSpecificThresholds: config.classSpecificThresholds ?? new Map(),
        }
    }

    /**
     * Calculate IOU between two bounding boxes
     */
    private calculateIOU(box1: BoundingBox, box2: BoundingBox): number {
        const xOverlap = Math.max(0, Math.min(box1.x2, box2.x2) - Math.max(box1.x1, box2.x1))
        const yOverlap = Math.max(0, Math.min(box1.y2, box2.y2) - Math.max(box1.y1, box2.y1))

        if (xOverlap <= 0 || yOverlap <= 0) return 0

        const overlapArea = xOverlap * yOverlap
        const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        const unionArea = area1 + area2 - overlapArea

        return overlapArea / unionArea
    }

    /**
     * Calculate center point of a box
     */
    private getCenter(box: BoundingBox): { x: number; y: number } {
        return { x: (box.x1 + box.x2) / 2, y: (box.y1 + box.y2) / 2 }
    }

    /**
     * Calculate distance between two points
     */
    private distance(p1: { x: number; y: number }, p2: { x: number; y: number }): number {
        const dx = p1.x - p2.x
        const dy = p1.y - p2.y
        return Math.sqrt(dx * dx + dy * dy)
    }

    /**
     * Get adaptive threshold based on class and tracking history
     */
    private getAdaptiveThreshold(classId: number, tracked?: TrackedObject): number {
        let threshold = this.config.iouThreshold

        // Use class-specific threshold if available
        if (this.config.classSpecificThresholds?.has(classId)) {
            threshold = this.config.classSpecificThresholds.get(classId)!.iou
        }

        // Adjust based on tracking confidence if adaptive mode is on
        if (this.config.adaptiveThresholds && tracked) {
            const confidenceFactor = tracked.avgConfidence
            const visibilityFactor = Math.min(tracked.visibleCount / this.config.minVisibleFrames, 1)

            // Lower threshold for high-confidence, stable tracks
            threshold *= 2 - confidenceFactor * visibilityFactor
        }

        return threshold
    }

    /**
     * Find best matching tracked object using multiple criteria
     */
    private findBestMatch(detection: Detection): TrackedObject | null {
        let bestMatch: TrackedObject | null = null
        let bestScore = 0

        const detBox: BoundingBox = { x1: detection.x1, y1: detection.y1, x2: detection.x2, y2: detection.y2 }
        const detCenter = this.getCenter(detBox)

        for (const tracked of this.trackedObjects.values()) {
            // Only match same class
            if (tracked.classId !== detection.classId) continue

            let score = 0
            let compareBox = tracked.smoothedBox

            // Use predicted position if velocity tracking is enabled
            if (this.config.useVelocityPrediction && tracked.predictedBox) {
                compareBox = tracked.predictedBox
            }

            // Calculate IOU score
            const iou = this.calculateIOU(detBox, compareBox)
            const adaptiveThreshold = this.getAdaptiveThreshold(detection.classId, tracked)

            if (iou < adaptiveThreshold) continue

            score += iou * (1 - this.config.velocityWeight - this.config.confidenceWeight)

            // Add velocity-based distance score
            if (this.config.velocityWeight > 0) {
                const predCenter = this.getCenter(compareBox)
                const dist = this.distance(detCenter, predCenter)
                const maxDist = Math.sqrt(Math.pow(detection.width, 2) + Math.pow(detection.height, 2))
                const distScore = 1 - Math.min(dist / maxDist, 1)
                score += distScore * this.config.velocityWeight
            }

            // Add confidence similarity score
            if (this.config.confidenceWeight > 0) {
                const confDiff = Math.abs(detection.confidence - tracked.avgConfidence)
                const confScore = 1 - confDiff
                score += confScore * this.config.confidenceWeight
            }

            if (score > bestScore) {
                bestScore = score
                bestMatch = tracked
            }
        }

        return bestMatch
    }

    /**
     * Update velocity estimation
     */
    private updateVelocity(tracked: TrackedObject, newBox: BoundingBox): void {
        const oldCenter = this.getCenter(tracked.smoothedBox)
        const newCenter = this.getCenter(newBox)

        const vx = newCenter.x - oldCenter.x
        const vy = newCenter.y - oldCenter.y

        const alpha = this.config.velocitySmoothingFactor
        tracked.velocity.vx = alpha * tracked.velocity.vx + (1 - alpha) * vx
        tracked.velocity.vy = alpha * tracked.velocity.vy + (1 - alpha) * vy
    }

    /**
     * Predict next position based on velocity
     */
    private predictNextPosition(tracked: TrackedObject): BoundingBox {
        const center = this.getCenter(tracked.smoothedBox)
        const predictedCenter = { x: center.x + tracked.velocity.vx, y: center.y + tracked.velocity.vy }

        const halfWidth = (tracked.smoothedBox.x2 - tracked.smoothedBox.x1) / 2
        const halfHeight = (tracked.smoothedBox.y2 - tracked.smoothedBox.y1) / 2

        return {
            x1: predictedCenter.x - halfWidth,
            y1: predictedCenter.y - halfHeight,
            x2: predictedCenter.x + halfWidth,
            y2: predictedCenter.y + halfHeight,
        }
    }

    /**
     * Smooth bounding box with exponential moving average
     */
    private smoothBox(current: Detection, tracked: TrackedObject): void {
        const alpha = this.config.boxSmoothingFactor
        const beta = 1 - alpha

        // Update velocity before smoothing
        this.updateVelocity(tracked, { x1: current.x1, y1: current.y1, x2: current.x2, y2: current.y2 })

        // Smooth the box
        tracked.smoothedBox.x1 = alpha * current.x1 + beta * tracked.smoothedBox.x1
        tracked.smoothedBox.y1 = alpha * current.y1 + beta * tracked.smoothedBox.y1
        tracked.smoothedBox.x2 = alpha * current.x2 + beta * tracked.smoothedBox.x2
        tracked.smoothedBox.y2 = alpha * current.y2 + beta * tracked.smoothedBox.y2

        // Update confidence tracking
        tracked.confidenceHistory.push(current.confidence)
        if (tracked.confidenceHistory.length > this.config.maxConfidenceHistory) {
            tracked.confidenceHistory.shift()
        }

        tracked.avgConfidence = tracked.confidenceHistory.reduce((a, b) => a + b) / tracked.confidenceHistory.length
        tracked.maxConfidence = Math.max(tracked.maxConfidence, current.confidence)

        // Update detection with smoothed values
        tracked.detection = {
            ...current,
            x1: tracked.smoothedBox.x1,
            y1: tracked.smoothedBox.y1,
            x2: tracked.smoothedBox.x2,
            y2: tracked.smoothedBox.y2,
            width: tracked.smoothedBox.x2 - tracked.smoothedBox.x1,
            height: tracked.smoothedBox.y2 - tracked.smoothedBox.y1,
        }
    }

    /**
     * Process detections with temporal filtering
     */
    processFrame(rawDetections: Detection[]): Array<Detection & Partial<TrackedObject>> {
        this.frameNumber++
        const matchedIds = new Set<string>()

        // Predict positions for next frame
        if (this.config.useVelocityPrediction) {
            for (const tracked of this.trackedObjects.values()) {
                tracked.predictedBox = this.predictNextPosition(tracked)
            }
        }

        // Match new detections to tracked objects
        for (const detection of rawDetections) {
            const tracked = this.findBestMatch(detection)

            if (tracked) {
                matchedIds.add(tracked.id)

                // Smooth the bounding box
                this.smoothBox(detection, tracked)

                // Update tracking state
                tracked.hiddenCount = 0
                tracked.lastSeen = this.frameNumber

                // Update visibility
                if (tracked.visibleCount < this.config.minVisibleFrames) {
                    tracked.visibleCount++
                    if (tracked.visibleCount >= this.config.minVisibleFrames) {
                        tracked.isVisible = true
                        console.log(
                            `Object ${tracked.id} (${tracked.className}) visible after ${
                                this.frameNumber - tracked.firstSeen
                            } frames`,
                        )
                    }
                }
            } else {
                // Create new tracked object
                const newId = `obj_${this.nextObjectId++}`
                const newTracked: TrackedObject = {
                    id: newId,
                    classId: detection.classId,
                    className: detection.className,
                    detection: detection,
                    visibleCount: 1,
                    hiddenCount: 0,
                    isVisible: false,
                    lastSeen: this.frameNumber,
                    firstSeen: this.frameNumber,
                    smoothedBox: { x1: detection.x1, y1: detection.y1, x2: detection.x2, y2: detection.y2 },
                    velocity: { vx: 0, vy: 0 },
                    avgConfidence: detection.confidence,
                    maxConfidence: detection.confidence,
                    confidenceHistory: [detection.confidence],
                }

                this.trackedObjects.set(newId, newTracked)
                matchedIds.add(newId)
            }
        }

        // Update unmatched objects
        const objectsToRemove: string[] = []

        for (const [id, tracked] of this.trackedObjects) {
            if (!matchedIds.has(id)) {
                tracked.hiddenCount++

                // Use velocity to predict position while hidden
                if (this.config.useVelocityPrediction && tracked.hiddenCount <= 2) {
                    const predicted = this.predictNextPosition(tracked)
                    tracked.smoothedBox = predicted
                    tracked.detection.x1 = predicted.x1
                    tracked.detection.y1 = predicted.y1
                    tracked.detection.x2 = predicted.x2
                    tracked.detection.y2 = predicted.y2
                }

                // Remove if hidden too long
                if (tracked.hiddenCount > this.config.maxHiddenFrames) {
                    objectsToRemove.push(id)
                    console.log(
                        `Removing ${id} (${tracked.className}) - hidden for ${
                            tracked.hiddenCount
                        } frames, confidence was ${tracked.avgConfidence.toFixed(2)}`,
                    )
                }
            }
        }

        // Remove dead objects
        objectsToRemove.forEach(id => this.trackedObjects.delete(id))

        // Return visible objects with metadata
        const visibleDetections: Array<Detection & Partial<TrackedObject>> = []

        for (const tracked of this.trackedObjects.values()) {
            if (tracked.isVisible) {
                visibleDetections.push({
                    ...tracked.detection,
                    id: tracked.id,
                    visibleFrames: tracked.visibleCount,
                    hiddenFrames: tracked.hiddenCount,
                    avgConfidence: tracked.avgConfidence,
                    velocity: { ...tracked.velocity },
                })
            }
        }

        return visibleDetections
    }

    getStats(): { frameNumber: number; totalTracked: number; visibleObjects: number; hiddenObjects: number } {
        const visible = Array.from(this.trackedObjects.values()).filter(t => t.isVisible).length

        return {
            frameNumber: this.frameNumber,
            totalTracked: this.trackedObjects.size,
            visibleObjects: visible,
            hiddenObjects: this.trackedObjects.size - visible,
        }
    }

    reset(): void {
        this.trackedObjects.clear()
        this.nextObjectId = 1
        this.frameNumber = 0
    }
}
