import { WebSocket, WebSocketServer } from 'ws'
import { EventEmitter } from 'events'

interface Client {
    ws: WebSocket
    id: string
    channels: Set<string>
    isAlive: boolean
}

interface Message {
    type: 'subscribe' | 'unsubscribe' | 'ping'
    channel?: string
}

export class WSServer extends EventEmitter {
    private server: WebSocketServer
    private clients = new Map<string, Client>()
    private heartbeatInterval: NodeJS.Timeout | null = null
    private readonly HEARTBEAT_INTERVAL = 30000

    constructor(port: number = 8080) {
        super()
        this.server = new WebSocketServer({ port, skipUTF8Validation: true, perMessageDeflate: false })
        this.setupServer()
    }

    private setupServer(): void {
        this.server.on('connection', (ws, request) => {
            const clientId = this.generateClientId()
            const client: Client = { ws, id: clientId, channels: new Set(), isAlive: true }

            this.clients.set(clientId, client)
            console.log(`Client ${clientId} connected from ${request.socket.remoteAddress}`)

            ws.on('pong', () => {
                const client = this.clients.get(clientId)
                if (client) client.isAlive = true
            })

            ws.on('message', data => {
                try {
                    const message: Message = JSON.parse(data.toString())
                    this.handleClientMessage(clientId, message)
                } catch (error) {
                    console.error(`Invalid message from client ${clientId}:`, error)
                }
            })

            ws.on('close', (code, reason) => {
                console.log(`Client ${clientId} disconnected: ${code} ${reason}`)
                this.clients.delete(clientId)
                this.emit('client-disconnected', clientId)
            })

            ws.on('error', error => {
                console.error(`Client ${clientId} error:`, error)
                this.clients.delete(clientId)
            })

            this.sendToClient(clientId, {
                type: 'welcome',
                clientId,
                availableChannels: ['video', 'metadata', 'ai', 'ai-debug'],
            })

            this.emit('client-connected', clientId)
        })

        // Start heartbeat
        this.heartbeatInterval = setInterval(() => {
            this.clients.forEach((client, clientId) => {
                if (!client.isAlive) {
                    console.log(`Client ${clientId} failed heartbeat, terminating`)
                    client.ws.terminate()
                    this.clients.delete(clientId)
                    return
                }

                client.isAlive = false
                client.ws.ping()
            })
        }, this.HEARTBEAT_INTERVAL)
    }

    private generateClientId(): string {
        return Math.random().toString(36).substring(2, 15)
    }

    private handleClientMessage(clientId: string, message: Message): void {
        const client = this.clients.get(clientId)
        if (!client) return

        switch (message.type) {
            case 'subscribe':
                if (message.channel) {
                    client.channels.add(message.channel)
                    console.log(`Client ${clientId} subscribed to: ${message.channel}`)
                    this.sendToClient(clientId, { type: 'subscribed', channel: message.channel })
                    this.emit('channel-subscribed', { clientId, channel: message.channel })
                }
                break

            case 'unsubscribe':
                if (message.channel) {
                    client.channels.delete(message.channel)
                    console.log(`Client ${clientId} unsubscribed from: ${message.channel}`)
                    this.sendToClient(clientId, { type: 'unsubscribed', channel: message.channel })
                    this.emit('channel-unsubscribed', { clientId, channel: message.channel })
                }
                break

            case 'ping':
                this.sendToClient(clientId, { type: 'pong' })
                break
        }
    }

    sendToClient(clientId: string, message: unknown): void {
        const client = this.clients.get(clientId)
        if (!client || client.ws.readyState !== WebSocket.OPEN) return

        try {
            client.ws.send(JSON.stringify(message))
        } catch (error) {
            console.error(`Failed to send to client ${clientId}:`, error)
            this.clients.delete(clientId)
        }
    }

    broadcastBinary(channel: string, data: Buffer, metadata?: unknown): void {
        const message = { channel, timestamp: Date.now(), metadata }

        const header = Buffer.from(JSON.stringify(message))
        const headerLength = Buffer.alloc(4)
        headerLength.writeUInt32BE(header.length, 0)

        const combined = Buffer.concat([headerLength, header, data])

        this.clients.forEach(client => {
            if (client.channels.has(channel) && client.ws.readyState === WebSocket.OPEN) {
                try {
                    client.ws.send(combined, { binary: true })
                } catch (error) {
                    console.error(`Failed to send binary to ${client.id}:`, error)
                    this.clients.delete(client.id)
                }
            }
        })
    }

    broadcast(channel: string, message: object): void {
        const payload = { channel, timestamp: Date.now(), ...message }

        this.clients.forEach(client => {
            if (client.channels.has(channel) && client.ws.readyState === WebSocket.OPEN) {
                try {
                    client.ws.send(JSON.stringify(payload))
                } catch (error) {
                    console.error(`Failed to send to ${client.id}:`, error)
                    this.clients.delete(client.id)
                }
            }
        })
    }

    getStats(): { clientCount: number; channels: string[] } {
        return {
            clientCount: this.clients.size,
            channels: Array.from(new Set(Array.from(this.clients.values()).flatMap(c => Array.from(c.channels)))),
        }
    }

    async stop(): Promise<void> {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval)
        }

        this.clients.forEach(client => {
            client.ws.close(1000, 'Server shutting down')
        })

        return new Promise(resolve => {
            this.server.close(() => {
                console.log('WebSocket server closed')
                resolve()
            })
        })
    }
}
