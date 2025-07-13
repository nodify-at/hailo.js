// detection.client.js - WebGL-powered WebSocket handler

class DetectionClient {
    constructor(config = {}) {
        this.wsUrl = config.wsUrl || 'ws://192.168.0.187:8080'
        this.onFrame = config.onFrame || (() => {})
        this.onDetections = config.onDetections || (() => {})
        this.onStats = config.onStats || (() => {})
        this.onStatusChange = config.onStatusChange || (() => {})

        this.ws = null
        this.reconnectAttempts = 0
        this.maxReconnectAttempts = 10
        this.reconnectDelay = 1000

        // Performance tracking
        this.frameCount = 0
        this.lastFrameTime = Date.now()
        this.localFps = 0

        // No frame dropping - process everything
        this.detectionHistory = new Map()

        this.connect()
    }

    connect() {
        this.onStatusChange('connecting')

        try {
            this.ws = new WebSocket(this.wsUrl)
            this.ws.binaryType = 'arraybuffer'

            this.ws.onopen = () => {
                console.log('✅ Connected to detection server')
                this.reconnectAttempts = 0
                this.onStatusChange('connected')

                // Subscribe to channels
                this.subscribe('video')
                this.subscribe('ai')
                this.subscribe('metadata')

                // Start ping
                this.ping()
            }

            this.ws.onmessage = event => {
                if (event.data instanceof ArrayBuffer) {
                    this.handleBinaryMessage(event.data)
                } else {
                    this.handleJsonMessage(event.data)
                }
            }

            this.ws.onclose = event => {
                console.log('❌ Disconnected:', event.code, event.reason)
                this.onStatusChange('disconnected')
                this.attemptReconnect()
            }

            this.ws.onerror = error => {
                console.error('WebSocket error:', error)
                this.onStatusChange('error')
            }
        } catch (error) {
            console.error('Failed to create WebSocket:', error)
            this.onStatusChange('error')
            this.attemptReconnect()
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached')
            this.onStatusChange('failed')
            return
        }

        this.reconnectAttempts++
        const delay = this.reconnectDelay * Math.min(this.reconnectAttempts, 5)

        console.log(`Reconnecting in ${delay}ms... (attempt ${this.reconnectAttempts})`)
        this.onStatusChange('reconnecting')

        setTimeout(() => this.connect(), delay)
    }

    subscribe(channel) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'subscribe', channel }))
        }
    }

    ping() {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'ping' }))
            setTimeout(() => this.ping(), 30000)
        }
    }

    handleBinaryMessage(data) {
        // Process immediately - no dropping
        const view = new DataView(data)
        const headerLength = view.getUint32(0)
        const headerBytes = new Uint8Array(data, 4, headerLength)
        const header = JSON.parse(new TextDecoder().decode(headerBytes))

        if (header.channel === 'video' && header.metadata?.type === 'jpeg') {
            const imageData = new Uint8Array(data, 4 + headerLength)
            const blob = new Blob([imageData], { type: 'image/jpeg' })

            // Update FPS counter
            this.frameCount++
            const now = Date.now()
            if (now - this.lastFrameTime >= 1000) {
                this.localFps = this.frameCount
                this.frameCount = 0
                this.lastFrameTime = now
            }

            this.onFrame({ blob, metadata: header.metadata, timestamp: header.timestamp, localFps: this.localFps })
        }
    }

    handleJsonMessage(data) {
        try {
            const message = JSON.parse(data)

            switch (message.channel) {
                case 'ai':
                    if (message.type === 'detections') {
                        // Process detections for smooth animations
                        const processed = this.processDetectionsForAnimation(message.detections)
                        this.onDetections({ ...message, detections: processed })
                    }
                    break

                case 'metadata':
                    if (message.type === 'system_stats') {
                        this.onStats(message)
                    }
                    break

                default:
                    if (message.type === 'welcome') {
                        console.log('Server info:', message)
                    }
            }
        } catch (error) {
            console.error('Failed to parse message:', error)
        }
    }

    processDetectionsForAnimation(detections) {
        const currentTime = Date.now()
        const activeDetections = new Map()

        // Track detections for smooth animations
        detections.forEach(detection => {
            const id = detection.id || `${detection.classId}_${Math.floor(detection.x1)}_${Math.floor(detection.y1)}`

            const existing = this.detectionHistory.get(id)
            detection.isNew = !existing
            detection.animationProgress = existing ? 1.0 : 0.0

            activeDetections.set(id, { detection, lastSeen: currentTime })
        })

        // Add fading detections
        const allDetections = [...detections]
        this.detectionHistory.forEach((value, id) => {
            if (!activeDetections.has(id) && currentTime - value.lastSeen < 300) {
                value.detection.isFading = true
                value.detection.animationProgress = 1.0 - (currentTime - value.lastSeen) / 300
                allDetections.push(value.detection)
            }
        })

        this.detectionHistory = activeDetections
        return allDetections
    }

    disconnect() {
        if (this.ws) {
            this.ws.close(1000, 'Client disconnect')
            this.ws = null
        }
    }

    getConnectionState() {
        if (!this.ws) return 'disconnected'
        switch (this.ws.readyState) {
            case WebSocket.CONNECTING:
                return 'connecting'
            case WebSocket.OPEN:
                return 'connected'
            case WebSocket.CLOSING:
                return 'closing'
            case WebSocket.CLOSED:
                return 'disconnected'
            default:
                return 'unknown'
        }
    }
}
