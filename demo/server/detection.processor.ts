import { Detection, HailoDevice, ModelInfo, parseNMS, resizeImage } from '../../lib/index.js'
import { EventEmitter } from 'events'

interface ProcessorConfig {
    modelPath: string
    confidenceThreshold: number
    jpegWidth: number
    jpegHeight: number
    rgbWidth: number
    rgbHeight: number
    modelInputSize: number
    processEveryNFrames: number
}

export class DetectionProcessor extends EventEmitter {
    private device: HailoDevice | null = null
    private modelInfo: ModelInfo | null = null
    private config: ProcessorConfig
    private frameCount = 0
    private processing = false
    private lastProcessTime = 0

    constructor(config: Partial<ProcessorConfig> = {}) {
        super()

        this.config = {
            modelPath: config.modelPath ?? 'yolov11x.hef',
            confidenceThreshold: config.confidenceThreshold ?? 0.5,
            jpegWidth: config.jpegWidth ?? 1280,
            jpegHeight: config.jpegHeight ?? 720,
            rgbWidth: config.rgbWidth ?? 640,
            rgbHeight: config.rgbHeight ?? 360,
            modelInputSize: config.modelInputSize ?? 640,
            processEveryNFrames: config.processEveryNFrames ?? 3,
        }
    }

    async initialize(): Promise<void> {
        try {
            const devices = await HailoDevice.scanDevices()
            console.log('Available Hailo devices:', devices)

            this.device = new HailoDevice({ debug: true })
            await this.device.loadModel(this.config.modelPath)

            this.modelInfo = this.device.getModelInfo()
            console.log('Model loaded successfully:', this.modelInfo)

            this.emit('initialized', this.modelInfo)
        } catch (error) {
            console.error('Failed to initialize detector:', error)
            throw error
        }
    }

    async processFrame(rgbData: Buffer): Promise<Detection[] | null> {
        this.frameCount++

        // Skip frames based on configuration
        if (this.frameCount % this.config.processEveryNFrames !== 0) {
            return null
        }

        // Skip if already processing
        if (this.processing) {
            return null
        }

        this.processing = true
        const startTime = Date.now()

        try {
            if (!this.device || !this.device.isReady() || !this.modelInfo) {
                throw new Error('Device not ready')
            }

            // Resize RGB to model input size with letterboxing
            const resized = await resizeImage(
                rgbData,
                this.config.rgbWidth,
                this.config.rgbHeight,
                this.config.modelInputSize,
                this.config.modelInputSize,
                this.frameCount === this.config.processEveryNFrames, // Log first resize
            )

            // Run inference
            const inputName = this.modelInfo.inputs[0]
            if (!inputName) {
                throw new Error('No input tensor found')
            }

            const outputs = await this.device.infer({ [inputName]: resized })
            if (!outputs) {
                console.warn('No outputs from inference')
                return null
            }

            // Parse NMS output with multi-stage coordinate transformation
            const nmsOutput = outputs['yolov11x/yolov8_nms_postprocess']
            if (!nmsOutput) {
                console.warn('No NMS output found')
                return null
            }

            const detections = parseNMS(nmsOutput, 80, {
                threshold: this.config.confidenceThreshold,
                jpegWidth: this.config.jpegWidth,
                jpegHeight: this.config.jpegHeight,
                rgbWidth: this.config.rgbWidth,
                rgbHeight: this.config.rgbHeight,
                modelInputSize: this.config.modelInputSize,
            })

            const processingTime = Date.now() - startTime
            this.lastProcessTime = processingTime

            this.emit('processed', { detections, frameNumber: this.frameCount, processingTime })

            return detections
        } catch (error) {
            console.error('Error processing frame:', error)
            this.emit('error', error)
            return null
        } finally {
            this.processing = false
        }
    }

    getStats(): {
        framesProcessed: number
        totalFrames: number
        lastProcessingTime: number
        isProcessing: boolean
        deviceReady: boolean
    } {
        return {
            framesProcessed: Math.floor(this.frameCount / this.config.processEveryNFrames),
            totalFrames: this.frameCount,
            lastProcessingTime: this.lastProcessTime,
            isProcessing: this.processing,
            deviceReady: this.device?.isReady() ?? false,
        }
    }

    async cleanup(): Promise<void> {
        if (this.device) {
            // Device cleanup if needed
            this.device = null
        }
        this.modelInfo = null
        this.removeAllListeners()
    }
}
