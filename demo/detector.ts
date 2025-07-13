import { HailoDevice, InferenceOutputs, ModelInfo } from '../lib/index.js'

export class Detector {
    private modelInfo: ModelInfo

    constructor(private readonly manager: HailoDevice) {
        this.modelInfo = this.manager.getModelInfo()
    }
    static async with(hef: string) {
        const devices = await HailoDevice.scanDevices()
        console.log(devices)

        const deviceManager = new HailoDevice({ debug: true })
        await deviceManager.loadModel(hef)

        return new Detector(deviceManager)
    }

    async detect(buffer: Uint8Array): Promise<InferenceOutputs | undefined> {
        if (!this.manager.isReady()) {
            console.error('Model is already loaded')
            return undefined
        }
        const inputName = this.modelInfo.inputs[0]!
        if (!inputName) {
            throw new Error('No input found')
        }
        return await this.manager.infer({ [inputName]: buffer })
    }
}
