import { AfMode, builder, Camera, ExposureMode } from '@nodify_at/picamera.js'
import { EventEmitter } from 'events'

export interface CameraConfig {
    width: number
    height: number
    rgbWidth: number
    rgbHeight: number
    quality: number
    fps: number
}

export interface FrameEvent {
    data: Buffer
    timestamp: number
    frameNumber: number
}

export class CameraStream extends EventEmitter {
    private readonly camera: Camera
    private readonly config: CameraConfig

    private isRunning = false
    private frameNumber = 0

    constructor(config: Partial<CameraConfig> = {}) {
        super()

        this.config = {
            width: config.width ?? 1280,
            height: config.height ?? 760,
            rgbWidth: config.rgbWidth ?? 640,
            rgbHeight: config.rgbHeight ?? 360,
            quality: config.quality ?? 70,
            fps: config.fps ?? 30,
        }

        this.camera = builder()
            .jpeg(this.config.width, this.config.height)
            .rgb(this.config.rgbWidth, this.config.rgbHeight)
            .raw()
            .quality(this.config.quality)
            .fps(this.config.fps)
            .focus(AfMode.AUTO)
            .exposure(ExposureMode.SHORT)
            .build()
    }

    async start(): Promise<void> {
        if (this.isRunning) {
            console.warn('Camera is already running')
            return
        }

        const started = this.camera.start()
        if (!started) {
            throw new Error('Failed to start camera')
        }

        this.isRunning = true
        this.frameNumber = 0
        console.log('Camera stream started', this.config)

        // Setup frame handlers
        this.camera.on('jpeg', event => {
            this.frameNumber++
            this.emit('jpeg', { data: event.data, timestamp: Date.now(), frameNumber: this.frameNumber } as FrameEvent)
        })

        this.camera.on('rgb', event => {
            this.emit('rgb', { data: event.data, timestamp: Date.now(), frameNumber: this.frameNumber } as FrameEvent)
        })
    }

    stop(): void {
        if (!this.isRunning) return

        this.isRunning = false
        this.camera.stop()
        this.removeAllListeners()
        console.log('Camera stream stopped')
    }

    getConfig(): CameraConfig {
        return { ...this.config }
    }

    isActive(): boolean {
        return this.isRunning
    }

    getFrameNumber(): number {
        return this.frameNumber
    }
}
