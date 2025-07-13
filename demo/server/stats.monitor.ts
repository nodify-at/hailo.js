import { EventEmitter } from 'events'

interface FrameStats {
    fps: number
    avgFps: number
    frameSize: number
    frameCount: number
}

interface ProcessingStats {
    inferenceTime: number
    avgInferenceTime: number
    detectionsPerFrame: number
    processedFrames: number
}

export interface TrackingStats {
    totalTracked: number
    visibleObjects: number
    hiddenObjects: number
    newObjects: number
    removedObjects: number
}

export interface SystemStats {
    frame: FrameStats
    processing: ProcessingStats
    tracking: TrackingStats
    network: { clientCount: number; channels: string[]; bytesPerSecond: number }
    timestamp: number
}

export class StatsMonitor extends EventEmitter {
    private stats: SystemStats
    private frameTimestamps: number[] = []
    private inferenceTimings: number[] = []
    private bytesSent = 0
    private lastBytesReset = Date.now()
    private readonly WINDOW_SIZE = 30 // Keep last 30 samples

    constructor() {
        super()

        this.stats = {
            frame: { fps: 0, avgFps: 0, frameSize: 0, frameCount: 0 },
            processing: { inferenceTime: 0, avgInferenceTime: 0, detectionsPerFrame: 0, processedFrames: 0 },
            tracking: { totalTracked: 0, visibleObjects: 0, hiddenObjects: 0, newObjects: 0, removedObjects: 0 },
            network: { clientCount: 0, channels: [], bytesPerSecond: 0 },
            timestamp: Date.now(),
        }
    }

    updateFrameStats(frameSize: number): void {
        const now = Date.now()
        this.frameTimestamps.push(now)

        // Keep only recent timestamps
        const cutoff = now - 1000
        this.frameTimestamps = this.frameTimestamps.filter(t => t > cutoff)

        // Calculate instantaneous FPS
        this.stats.frame.fps = this.frameTimestamps.length
        this.stats.frame.frameSize = frameSize
        this.stats.frame.frameCount++

        // Calculate average FPS over window
        if (this.frameTimestamps.length > 1) {
            const duration = (now - this.frameTimestamps[0]!) / 1000
            this.stats.frame.avgFps = this.frameTimestamps.length / duration
        }

        this.bytesSent += frameSize
    }

    updateProcessingStats(inferenceTime: number, detectionCount: number): void {
        this.inferenceTimings.push(inferenceTime)

        // Keep window size
        if (this.inferenceTimings.length > this.WINDOW_SIZE) {
            this.inferenceTimings.shift()
        }

        this.stats.processing.inferenceTime = inferenceTime
        this.stats.processing.avgInferenceTime =
            this.inferenceTimings.reduce((a, b) => a + b, 0) / this.inferenceTimings.length

        this.stats.processing.detectionsPerFrame = detectionCount
        this.stats.processing.processedFrames++
    }

    updateTrackingStats(stats: Partial<TrackingStats>): void {
        Object.assign(this.stats.tracking, stats)
    }

    updateNetworkStats(clientCount: number, channels: string[]): void {
        this.stats.network.clientCount = clientCount
        this.stats.network.channels = channels

        // Calculate bytes per second
        const now = Date.now()
        const elapsed = (now - this.lastBytesReset) / 1000

        if (elapsed >= 1) {
            this.stats.network.bytesPerSecond = Math.round(this.bytesSent / elapsed)
            this.bytesSent = 0
            this.lastBytesReset = now
        }
    }

    getStats(): SystemStats {
        this.stats.timestamp = Date.now()
        return { ...this.stats }
    }

    /**
     * Get formatted stats for logging
     */
    getFormattedStats(): string {
        const s = this.stats
        return (
            `FPS: ${s.frame.fps}/${s.frame.avgFps.toFixed(1)} | ` +
            `Inference: ${s.processing.inferenceTime}ms (avg: ${s.processing.avgInferenceTime.toFixed(1)}ms) | ` +
            `Tracked: ${s.tracking.totalTracked} (${s.tracking.visibleObjects} visible) | ` +
            `Clients: ${s.network.clientCount} | ` +
            `Bandwidth: ${(s.network.bytesPerSecond / 1024 / 1024).toFixed(2)} MB/s`
        )
    }

    reset(): void {
        this.frameTimestamps = []
        this.inferenceTimings = []
        this.bytesSent = 0
        this.lastBytesReset = Date.now()

        // Reset counters but keep current values
        this.stats.frame.frameCount = 0
        this.stats.processing.processedFrames = 0
    }
}
