import { WSServer } from './server/websocket.server.js'
import { CameraStream, FrameEvent } from './server/camera.stream.js'
import { TemporalFilter } from './server/temporal.filter.js'
import { DetectionProcessor } from './server/detection.processor.js'
import { StatsMonitor } from './server/stats.monitor.js'
import { Detection } from '../lib/index.js'

// Optimized class-specific tracking thresholds based on object characteristics
const OPTIMIZED_CLASS_THRESHOLDS = new Map([
    [1, { iou: 0.15, confidence: 0.6 }], // person - more movement tolerance
    [2, { iou: 0.08, confidence: 0.5 }], // bicycle - moderate tracking
    [3, { iou: 0.1, confidence: 0.55 }], // car - balanced tracking
    [4, { iou: 0.1, confidence: 0.55 }], // motorcycle
    [17, { iou: 0.12, confidence: 0.65 }], // dog - higher movement tolerance
    [16, { iou: 0.12, confidence: 0.65 }], // cat - similar to dog
    [62, { iou: 0.05, confidence: 0.7 }], // laptop - minimal movement
    [63, { iou: 0.06, confidence: 0.7 }], // mouse - small object
    [67, { iou: 0.05, confidence: 0.7 }], // cell phone - static tracking
])

export class LiveDetectionServer {
    private wsServer: WSServer
    private camera: CameraStream
    private detector: DetectionProcessor
    private temporalFilter: TemporalFilter
    private statsMonitor: StatsMonitor

    private isRunning = false
    private currentDetections: Detection[] = []
    private statsInterval: NodeJS.Timeout | null = null

    // Expose minimal configuration
    private readonly port: number
    private readonly modelPath: string

    constructor(options: { port?: number; modelPath?: string } = {}) {
        this.port = options.port ?? 8080
        this.modelPath = options.modelPath ?? 'yolov11x.hef'

        // Initialize all components with optimized settings
        this.wsServer = new WSServer(this.port)

        // Camera optimized for balance between quality and performance
        this.camera = new CameraStream({
            width: 1280, // HD output for good detail
            height: 720,
            rgbWidth: 640, // Efficient processing size
            rgbHeight: 360,
            quality: 75, // Good quality without excessive bandwidth
            fps: 30, // Smooth video
        })

        // Detection processor with optimal frame processing
        this.detector = new DetectionProcessor({
            modelPath: this.modelPath,
            confidenceThreshold: 0.45, // Balanced threshold
            jpegWidth: 1280,
            jpegHeight: 720,
            rgbWidth: 640,
            rgbHeight: 360,
            modelInputSize: 640,
            processEveryNFrames: 2, // Process every other frame for responsiveness
        })

        // Temporal filter with advanced features enabled
        this.temporalFilter = new TemporalFilter({
            iouThreshold: 0.1,
            velocityWeight: 0.25, // Increased for better motion tracking
            confidenceWeight: 0.15, // Consider confidence in matching
            boxSmoothingFactor: 0.7, // Smooth but responsive
            velocitySmoothingFactor: 0.85, // Stable velocity estimation
            minVisibleFrames: 5, // Quick appearance
            maxHiddenFrames: 5, // Tolerance for brief occlusions
            useVelocityPrediction: true, // Always use velocity prediction
            adaptiveThresholds: true, // Dynamic thresholds
            maxConfidenceHistory: 15, // Good history for averaging
            classSpecificThresholds: OPTIMIZED_CLASS_THRESHOLDS,
        })

        this.statsMonitor = new StatsMonitor()
        this.setupEventHandlers()
    }

    private setupEventHandlers(): void {
        // Camera event handlers with optimized flow
        this.camera.on('jpeg', (event: FrameEvent) => {
            if (!this.isRunning) return

            this.statsMonitor.updateFrameStats(event.data.length)

            // Broadcast video with metadata
            this.wsServer.broadcastBinary('video', event.data, {
                type: 'jpeg',
                width: 1280,
                height: 720,
                timestamp: event.timestamp,
                frameNumber: event.frameNumber,
            })

            // Always send current detections for smooth overlay
            if (this.currentDetections.length > 0) {
                this.wsServer.broadcast('ai', {
                    type: 'detections',
                    detections: this.currentDetections,
                    frameNumber: event.frameNumber,
                    smooth: true, // Indicate these are smoothed detections
                })
            }
        })

        this.camera.on('rgb', async (event: FrameEvent) => {
            if (!this.isRunning) return
            await this.detector.processFrame(event.data)
        })

        // Detection processor with comprehensive handling
        this.detector.on('processed', ({ detections, frameNumber, processingTime }) => {
            // Update processing stats
            this.statsMonitor.updateProcessingStats(processingTime, detections.length)

            // Apply temporal filtering
            const filteredDetections = this.temporalFilter.processFrame(detections)
            this.currentDetections = filteredDetections

            // Get comprehensive tracking stats
            const trackingStats = this.temporalFilter.getStats()
            this.statsMonitor.updateTrackingStats({
                totalTracked: trackingStats.totalTracked,
                visibleObjects: trackingStats.visibleObjects,
                hiddenObjects: trackingStats.hiddenObjects,
            })

            // Broadcast enriched detection data
            this.wsServer.broadcast('ai', {
                type: 'detections',
                detections: filteredDetections,
                frameNumber,
                stats: {
                    raw: detections.length,
                    filtered: filteredDetections.length,
                    tracked: trackingStats.totalTracked,
                    visible: trackingStats.visibleObjects,
                    processingMs: processingTime,
                },
            })

            // Detailed debug information for development
            if (this.hasDebugClients()) {
                const debugData = this.createDebugData(
                    detections,
                    filteredDetections,
                    trackingStats,
                    processingTime,
                    frameNumber,
                )
                this.wsServer.broadcast('ai-debug', debugData)
            }
        })

        // Network event handlers
        this.wsServer.on('client-connected', (clientId: string) => {
            console.log(`[Client] Connected: ${clientId}`)
            this.updateNetworkStats()

            // Send initial state to new client
            this.wsServer.sendToClient(clientId, {
                type: 'server_info',
                version: '2.0',
                capabilities: { detection: true, tracking: true, velocityPrediction: true, adaptiveFiltering: true },
                model: this.modelPath,
            })
        })

        this.wsServer.on('client-disconnected', (clientId: string) => {
            console.log(`[Client] Disconnected: ${clientId}`)
            this.updateNetworkStats()
        })

        this.wsServer.on('channel-subscribed', ({ clientId, channel }) => {
            console.log(`[Client] ${clientId} → subscribed to ${channel}`)
        })
    }

    private createDebugData(
        raw: Detection[],
        filtered: Detection[],
        tracking: any,
        processingTime: number,
        frameNumber: number,
    ): object {
        // Create detailed debug information
        const classDistribution = new Map<string, number>()
        filtered.forEach(d => {
            classDistribution.set(d.className, (classDistribution.get(d.className) || 0) + 1)
        })

        return {
            type: 'tracking_debug',
            frameNumber,
            processingTime,
            detections: { raw: raw.length, filtered: filtered.length, suppressed: raw.length - filtered.length },
            tracking: {
                total: tracking.totalTracked,
                visible: tracking.visibleObjects,
                hidden: tracking.hiddenObjects,
                framesSinceLastNew: frameNumber - tracking.frameNumber,
            },
            classes: Array.from(classDistribution.entries()).map(([name, count]) => ({ name, count })),
            performance: {
                processingMs: processingTime,
                fps: this.statsMonitor.getStats().frame.fps,
                avgInferenceMs: this.statsMonitor.getStats().processing.avgInferenceTime,
            },
        }
    }

    private hasDebugClients(): boolean {
        // Check if any clients are subscribed to debug channel
        const stats = this.wsServer.getStats()
        return stats.channels.includes('ai-debug')
    }

    private updateNetworkStats(): void {
        const stats = this.wsServer.getStats()
        this.statsMonitor.updateNetworkStats(stats.clientCount, stats.channels)
    }

    private startStatsReporting(): void {
        // Single interval for all stats reporting
        this.statsInterval = setInterval(() => {
            const stats = this.statsMonitor.getStats()

            // Console logging - concise format
            console.log(`[Stats] ${this.statsMonitor.getFormattedStats()}`)

            // Broadcast comprehensive stats to clients
            this.wsServer.broadcast('metadata', {
                type: 'system_stats',
                ...stats,
                server: {
                    uptime: process.uptime(),
                    memory: process.memoryUsage().heapUsed / 1024 / 1024, // MB
                    model: this.modelPath,
                },
            })

            // Performance warnings
            if (stats.processing.avgInferenceTime > 50) {
                console.warn('[Perf] High inference time:', stats.processing.avgInferenceTime, 'ms')
            }
            if (stats.frame.fps < 25) {
                console.warn('[Perf] Low FPS:', stats.frame.fps)
            }
        }, 1000) // Update every second
    }

    async start(): Promise<void> {
        if (this.isRunning) {
            console.warn('Server is already running')
            return
        }

        console.log('═══════════════════════════════════════════')
        console.log('  Live Detection Server v2.0')
        console.log('═══════════════════════════════════════════')

        try {
            // Initialize detector first
            console.log('[Init] Loading AI model...')
            await this.detector.initialize()

            // Start camera
            console.log('[Init] Starting camera stream...')
            await this.camera.start()

            // Start stats reporting
            this.startStatsReporting()

            this.isRunning = true

            console.log('═══════════════════════════════════════════')
            console.log(`[Ready] WebSocket: ws://localhost:${this.port}`)
            console.log('[Ready] Model:', this.modelPath)
            console.log('[Ready] Stream: 1280x720 @ 30fps')
            console.log('[Ready] Features: Velocity prediction, Adaptive filtering')
            console.log('═══════════════════════════════════════════')
        } catch (error) {
            console.error('[Error] Failed to start:', error)
            await this.stop()
            throw error
        }
    }

    async stop(): Promise<void> {
        if (!this.isRunning && !this.statsInterval) {
            return
        }

        console.log('\n[Shutdown] Stopping server...')
        this.isRunning = false

        // Clear stats reporting
        if (this.statsInterval) {
            clearInterval(this.statsInterval)
            this.statsInterval = null
        }

        // Stop components in order
        this.camera.stop()
        await this.detector.cleanup()
        await this.wsServer.stop()

        // Reset state
        this.temporalFilter.reset()
        this.statsMonitor.reset()
        this.currentDetections = []

        console.log('[Shutdown] Server stopped successfully')
    }

    // Minimal public API
    getPort(): number {
        return this.port
    }

    isActive(): boolean {
        return this.isRunning
    }

    getClientCount(): number {
        return this.wsServer.getStats().clientCount
    }
}

// Direct execution
if (import.meta.url === `file://${process.argv[1]}`) {
    const server = new LiveDetectionServer({
        port: parseInt(process.env['WS_PORT'] || '8080'),
        modelPath: process.env['MODEL_PATH'] || 'yolov11x.hef',
    })

    // Graceful shutdown handlers
    const shutdown = async (signal: string): Promise<never> => {
        console.log(`\n[Signal] Received ${signal}`)
        await server.stop()
        process.exit(0)
    }

    process.on('SIGINT', () => shutdown('SIGINT'))
    process.on('SIGTERM', () => shutdown('SIGTERM'))
    process.on('SIGABRT', () => shutdown('SIGABRT'))

    // Handle uncaught errors
    process.on('uncaughtException', async (error: Error) => {
        console.error('[Fatal] Uncaught exception:', error)
        void server.stop()
        process.exit(0)
    })

    process.on('unhandledRejection', async reason => {
        console.error('[Fatal] Unhandled rejection:', reason)
        await server.stop()
        process.exit(1)
    })

    // Start server
    server.start().catch(error => {
        console.error('[Fatal] Failed to start:', error)
        process.exit(1)
    })
}
