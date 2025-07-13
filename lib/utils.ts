import sharp from 'sharp'
import { join } from 'path'

/**
 * Optimized resize using Sharp - much faster than custom implementation
 */
export async function resizeImage(
    buffer: Uint8Array,
    srcWidth: number,
    srcHeight: number,
    dstWidth: number,
    dstHeight: number,
    save = false,
): Promise<Uint8Array> {
    if (save) {
        await sharp(buffer, {
            raw: {
                width: srcWidth,
                height: srcHeight,
                channels: 3, // RGB
            },
        })
            .png() // Output as a raw RGB buffer
            .toFile(join(process.cwd(), 'output.png'))
        console.log('Saved output.png')
    }
    return await sharp(buffer, {
        raw: {
            width: srcWidth,
            height: srcHeight,
            channels: 3, // RGB
        },
    })
        .resize(dstWidth, dstHeight, {
            fit: 'contain', // 'contain' creates letterboxing, not 'cover'
            background: { r: 0, g: 0, b: 0 }, // Black letterbox background
        })
        .raw() // Output as a raw RGB buffer
        .toBuffer()
}
