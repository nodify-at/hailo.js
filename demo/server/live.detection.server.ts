import { WSServer } from './websocket.server.js'
import { CameraConfig, CameraStream, FrameEvent } from './camera.stream.js'
import { TemporalFilter } from './temporal.filter.js'
import { DetectionProcessor } from './detection.processor.js'
import { StatsMonitor } from './stats.monitor.js'
import { Detection } from '../../lib/index.js'

interface ServerConfig {
    websocket: { port: number }
    camera: { width: number; height: number; rgbWidth: number; rgbHeight: number; quality: number; fps: number }
    detection: { modelPath: string; confidenceThreshold: number; processEveryNFrames: number }
    temporal: {
        iouThreshold: number
        minVisibleFrames: number
        maxHiddenFrames: number
        useVelocityPrediction: boolean
    }
    stats: { logInterval: number; broadcastInterval: number }
}

export class LiveDetectionServer {
    private wsServer: WSServer
    private camera: CameraStream
    private detector: DetectionProcessor
    private temporalFilter: TemporalFilter
    private statsMonitor: StatsMonitor

    private config: ServerConfig
    private isRunning = false
    private currentDetections: Detection[] = []
    private intervals: NodeJS.Timeout[] = []

    constructor(config: Partial<ServerConfig> = {}) {
        this.config = {
            websocket: { port: config.websocket?.port ?? 8080 },
            camera: {
                width: config.camera?.width ?? 1280,
                height: config.camera?.height ?? 720,
                rgbWidth: config.camera?.rgbWidth ?? 640,
                rgbHeight: config.camera?.rgbHeight ?? 360,
                quality: config.camera?.quality ?? 70,
                fps: config.camera?.fps ?? 30,
            },
            detection: {
                modelPath: config.detection?.modelPath ?? 'yolov11x.hef',
                confidenceThreshold: config.detection?.confidenceThreshold ?? 0.5,
                processEveryNFrames: config.detection?.processEveryNFrames ?? 3,
            },
            temporal: {
                iouThreshold: config.temporal?.iouThreshold ?? 0.1,
                minVisibleFrames: config.temporal?.minVisibleFrames ?? 6,
                maxHiddenFrames: config.temporal?.maxHiddenFrames ?? 3,
                useVelocityPrediction: config.temporal?.useVelocityPrediction ?? true,
            },
            stats: {
                logInterval: config.stats?.logInterval ?? 5000,
                broadcastInterval: config.stats?.broadcastInterval ?? 1000,
            },
        }

        // Initialize components
        this.wsServer = new WSServer(this.config.websocket.port)
        this.camera = new CameraStream(this.config.camera)
        this.detector = new DetectionProcessor({
            ...this.config.detection,
            jpegWidth: this.config.camera.width,
            jpegHeight: this.config.camera.height,
            rgbWidth: this.config.camera.rgbWidth,
            rgbHeight: this.config.camera.rgbHeight,
        })
        this.temporalFilter = new TemporalFilter(this.config.temporal)
        this.statsMonitor = new StatsMonitor()

        this.setupEventHandlers()
    }

    private setupEventHandlers(): void {
        // Camera frame handlers
        this.camera.on('jpeg', (event: FrameEvent) => {
            this.handleJpegFrame(event)
        })

        this.camera.on('rgb', async (event: FrameEvent) => {
            await this.handleRgbFrame(event)
        })

        // Detection processor handlers
        this.detector.on('processed', ({ detections, frameNumber, processingTime }) => {
            this.statsMonitor.updateProcessingStats(processingTime, detections.length)

            // Apply temporal filtering
            const filteredDetections = this.temporalFilter.processFrame(detections)
            this.currentDetections = filteredDetections

            // Update tracking stats
            const trackingStats = this.temporalFilter.getStats()
            this.statsMonitor.updateTrackingStats({
                totalTracked: trackingStats.totalTracked,
                visibleObjects: trackingStats.visibleObjects,
                hiddenObjects: trackingStats.hiddenObjects,
            })

            // Broadcast detections
            this.wsServer.broadcast('ai', {
                type: 'detections',
                detections: filteredDetections,
                frameNumber,
                stats: { raw: detections.length, filtered: filteredDetections.length, ...trackingStats },
            })

            // Debug info
            const debugInfo = {
                type: 'tracking_debug',
                frameNumber,
                processingTime,
                rawDetections: detections.length,
                filteredDetections: filteredDetections.length,
                tracking: trackingStats,
            }
            this.wsServer.broadcast('ai-debug', debugInfo)
        })

        // WebSocket server handlers
        this.wsServer.on('client-connected', (clientId: string) => {
            console.log(`Client connected: ${clientId}`)
            this.updateNetworkStats()
        })

        this.wsServer.on('client-disconnected', (clientId: string) => {
            console.log(`Client disconnected: ${clientId}`)
            this.updateNetworkStats()
        })

        this.wsServer.on('channel-subscribed', ({ clientId, channel }) => {
            console.log(`Client ${clientId} subscribed to ${channel}`)
        })
    }

    private handleJpegFrame(event: FrameEvent): void {
        if (!this.isRunning) return

        this.statsMonitor.updateFrameStats(event.data.length)

        // Broadcast JPEG frame
        this.wsServer.broadcastBinary('video', event.data, {
            type: 'jpeg',
            width: this.config.camera.width,
            height: this.config.camera.height,
            timestamp: event.timestamp,
            frameNumber: event.frameNumber,
        })

        // Always send current detections with video frame for smooth display
        if (this.currentDetections.length > 0) {
            this.wsServer.broadcast('ai', {
                type: 'detections',
                detections: this.currentDetections,
                frameNumber: event.frameNumber,
            })
        }
    }

    private async handleRgbFrame(event: FrameEvent): Promise<void> {
        if (!this.isRunning) return

        try {
            await this.detector.processFrame(event.data)
        } catch (error) {
            console.error('Error processing RGB frame:', error)
        }
    }

    private updateNetworkStats(): void {
        const stats = this.wsServer.getStats()
        this.statsMonitor.updateNetworkStats(stats.clientCount, stats.channels)
    }

    private startStatsReporting(): void {
        // Console logging
        const logInterval = setInterval(() => {
            console.log(this.statsMonitor.getFormattedStats())
            console.log(`Detector: ${JSON.stringify(this.detector.getStats())}`)
        }, this.config.stats.logInterval)
        this.intervals.push(logInterval)

        // WebSocket broadcast
        const broadcastInterval = setInterval(() => {
            const stats = this.statsMonitor.getStats()
            this.wsServer.broadcast('metadata', { type: 'system_stats', ...stats })
        }, this.config.stats.broadcastInterval)
        this.intervals.push(broadcastInterval)
    }

    async start(): Promise<void> {
        if (this.isRunning) {
            console.warn('Server is already running')
            return
        }

        console.log('Starting Live Detection Server...')

        try {
            // Initialize detector
            await this.detector.initialize()

            // Start camera
            await this.camera.start()

            // Start stats reporting
            this.startStatsReporting()

            this.isRunning = true
            console.log('Live Detection Server started successfully')
            console.log(`WebSocket server listening on port ${this.config.websocket.port}`)
            console.log(
                `Camera: ${this.config.camera.width}x${this.config.camera.height} @ ${this.config.camera.fps}fps`,
            )
            console.log(`Model: ${this.config.detection.modelPath}`)
        } catch (error) {
            console.error('Failed to start server:', error)
            await this.stop()
            throw error
        }
    }

    async stop(): Promise<void> {
        if (!this.isRunning && this.intervals.length === 0) {
            return
        }

        console.log('Stopping Live Detection Server...')
        this.isRunning = false

        // Clear intervals
        this.intervals.forEach(interval => clearInterval(interval))
        this.intervals = []

        // Stop components
        this.camera.stop()
        await this.detector.cleanup()
        await this.wsServer.stop()

        // Reset filters and stats
        this.temporalFilter.reset()
        this.statsMonitor.reset()
        this.currentDetections = []

        console.log('Live Detection Server stopped')
    }

    getStats(): {
        isRunning: boolean
        camera: CameraConfig
        detector: ReturnType<DetectionProcessor['getStats']>
        temporal: ReturnType<TemporalFilter['getStats']>
        system: ReturnType<StatsMonitor['getStats']>
        websocket: ReturnType<WSServer['getStats']>
    } {
        return {
            isRunning: this.isRunning,
            camera: this.camera.getConfig(),
            detector: this.detector.getStats(),
            temporal: this.temporalFilter.getStats(),
            system: this.statsMonitor.getStats(),
            websocket: this.wsServer.getStats(),
        }
    }
}

// Main entry point
if (import.meta.url === `file://${process.argv[1]}`) {
    const server = new LiveDetectionServer({
        websocket: { port: 8080 },
        camera: {
            width: 1280,
            height: 720, // Corrected from 760 to 720 to match detection_utils
            rgbWidth: 640,
            rgbHeight: 360,
            quality: 70,
            fps: 30,
        },
        detection: { modelPath: 'yolov11x.hef', confidenceThreshold: 0.5, processEveryNFrames: 3 },
        temporal: { iouThreshold: 0.1, minVisibleFrames: 6, maxHiddenFrames: 3, useVelocityPrediction: true },
        stats: { logInterval: 5000, broadcastInterval: 1000 },
    })

    // Handle a graceful shutdown
    const shutdown = async (): Promise<never> => {
        console.log('\nShutting down...')
        await server.stop()
        process.exit(0)
    }

    process.on('SIGINT', shutdown)
    process.on('SIGTERM', shutdown)
    process.on('SIGABRT', shutdown)

    // Start server
    server.start().catch(error => {
        console.error('Failed to start server:', error)
        process.exit(1)
    })
}
