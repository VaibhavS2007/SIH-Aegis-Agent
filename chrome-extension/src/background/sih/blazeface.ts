import * as ort from 'onnxruntime-web/webgpu';

export interface FaceRegion {
  /** Pixel coordinates in the source image. */
  bbox: [number, number, number, number];
  confidence: number;
}

const INPUT_SIZE = 128;
const DEFAULT_THRESHOLD = 0.65;

let sessionPromise: Promise<ort.InferenceSession | null> | null = null;

function modelUrl(): string {
  const configured = import.meta.env.VITE_SIH_BLAZEFACE_MODEL_URL as string | undefined;
  if (configured) {
    if (configured.startsWith('/') && typeof chrome !== 'undefined' && chrome.runtime?.getURL) {
      return chrome.runtime.getURL(configured.slice(1));
    }
    return configured;
  }
  return typeof chrome !== 'undefined' && chrome.runtime?.getURL
    ? chrome.runtime.getURL('models/blazeface.onnx')
    : 'models/blazeface.onnx';
}

async function createSession(): Promise<ort.InferenceSession | null> {
  try {
    const providers: ort.InferenceSession.ExecutionProviderConfig[] =
      typeof navigator !== 'undefined' && 'gpu' in navigator ? ['webgpu', 'wasm'] : ['wasm'];

    // Keep ORT's WASM assets inside the extension when CPU fallback is used.
    if (typeof chrome !== 'undefined' && chrome.runtime?.getURL) {
      ort.env.wasm.wasmPaths = chrome.runtime.getURL('');
    }

    return await ort.InferenceSession.create(modelUrl(), {
      executionProviders: providers,
      graphOptimizationLevel: 'all',
    });
  } catch {
    // A missing model is a safe, expected state during development. The
    // screenshot remains protected by DOM/PII masking and is not rejected.
    return null;
  }
}

async function getSession(): Promise<ort.InferenceSession | null> {
  sessionPromise ??= createSession();
  return sessionPromise;
}

function outputArray(value: ort.Tensor | undefined): Float32Array | number[] | null {
  if (!value?.data) return null;
  return value.data as Float32Array | number[];
}

/**
 * Parse the common standalone BlazeFace ONNX output convention:
 * boxes [N,4] (x1,y1,x2,y2) and scores [N]. Models that emit extra values
 * are handled by taking the first four box values and first score value.
 */
function parseDetections(
  outputs: Record<string, ort.Tensor>,
  width: number,
  height: number,
): FaceRegion[] {
  const tensors = Object.values(outputs);
  const boxTensor = tensors.find(t => (t.dims.length >= 2 && t.dims[t.dims.length - 1] >= 4));
  const scoreTensor = tensors.find(t => t !== boxTensor && t.data.length >= 1);
  const boxes = outputArray(boxTensor);
  const scores = outputArray(scoreTensor);
  if (!boxes || !scores) return [];

  const threshold = Number(import.meta.env.VITE_SIH_BLAZEFACE_THRESHOLD ?? DEFAULT_THRESHOLD);
  const boxStride = boxTensor!.dims[boxTensor!.dims.length - 1];
  const count = Math.min(Math.floor(boxes.length / boxStride), scores.length);
  const regions: FaceRegion[] = [];

  for (let i = 0; i < count; i += 1) {
    const confidence = Number(scores[i]);
    if (!Number.isFinite(confidence) || confidence < threshold) continue;
    const offset = i * boxStride;
    // The Hugging Face standalone BlazeFace export stores detections as
    // [top_y, top_x, bottom_y, bottom_x, landmarks...]. Simpler exports use
    // [x1, y1, x2, y2], so support both layouts.
    const values = boxStride >= 16
      ? [Number(boxes[offset + 1]), Number(boxes[offset]), Number(boxes[offset + 3]), Number(boxes[offset + 2])]
      : [Number(boxes[offset]), Number(boxes[offset + 1]), Number(boxes[offset + 2]), Number(boxes[offset + 3])];
    if (values.some(value => !Number.isFinite(value))) continue;

    // Accept either normalized coordinates or pixel coordinates.
    const normalized = values.every(value => value >= -1 && value <= 1);
    const scaleX = normalized ? width : 1;
    const scaleY = normalized ? height : 1;
    const [x1, y1, x2, y2] = [values[0] * scaleX, values[1] * scaleY, values[2] * scaleX, values[3] * scaleY];
    const left = Math.max(0, Math.min(x1, x2));
    const top = Math.max(0, Math.min(y1, y2));
    const right = Math.min(width, Math.max(x1, x2));
    const bottom = Math.min(height, Math.max(y1, y2));
    if (right - left > 2 && bottom - top > 2) {
      regions.push({ bbox: [left, top, right - left, bottom - top], confidence });
    }
  }
  return regions;
}

export async function detectFaces(bitmap: ImageBitmap): Promise<FaceRegion[]> {
  const session = await getSession();
  if (!session) return [];

  const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) return [];
  context.drawImage(bitmap, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const pixels = context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  // BlazeFace ONNX exports use NCHW layout: [1, 3, 128, 128].
  const plane = INPUT_SIZE * INPUT_SIZE;
  const input = new Float32Array(plane * 3);
  for (let i = 0; i < plane; i += 1) {
    input[i] = pixels[i * 4] / 255;
    input[plane + i] = pixels[i * 4 + 1] / 255;
    input[plane * 2 + i] = pixels[i * 4 + 2] / 255;
  }

  const feeds: Record<string, ort.Tensor> = {};
  for (const name of session.inputNames) {
    if (/image|input/i.test(name)) {
      feeds[name] = new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    } else if (/conf|threshold/i.test(name)) {
      feeds[name] = new ort.Tensor('float32', new Float32Array([Number(import.meta.env.VITE_SIH_BLAZEFACE_THRESHOLD ?? DEFAULT_THRESHOLD)]), [1]);
    } else if (/max.*det/i.test(name)) {
      feeds[name] = new ort.Tensor('int64', new BigInt64Array([25n]), [1]);
    } else if (/iou/i.test(name)) {
      feeds[name] = new ort.Tensor('float32', new Float32Array([0.3]), [1]);
    }
  }
  if (Object.keys(feeds).length !== session.inputNames.length) return [];
  const outputs = await session.run(feeds);
  return parseDetections(outputs, bitmap.width, bitmap.height);
}

async function blobToBase64(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

/** Blur detected faces in-memory before the screenshot is sent to the VLM. */
export async function redactFacesFromBase64(base64: string): Promise<string> {
  if (typeof createImageBitmap !== 'function' || typeof OffscreenCanvas === 'undefined') {
    throw new Error('SIH image APIs are unavailable; refusing visual egress');
  }
  try {
    // Do not allow a visual payload to leave the browser when the configured
    // face model is unavailable. DOM/text-only tasks can still run, but vision
    // tasks fail closed instead of silently sending unredacted pixels.
    if (!(await getSession())) {
      throw new Error('SIH face-redaction model is unavailable; refusing visual egress');
    }
    const bytes = Uint8Array.from(atob(base64), character => character.charCodeAt(0));
    const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }));
    const regions = await detectFaces(bitmap);
    if (regions.length === 0) {
      bitmap.close();
      return base64;
    }

    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext('2d');
    if (!context) {
      bitmap.close();
      return base64;
    }
    context.drawImage(bitmap, 0, 0);
    context.filter = 'blur(18px)';
    for (const region of regions) {
      const [x, y, width, height] = region.bbox;
      const padding = Math.max(width, height) * 0.2;
      context.drawImage(bitmap, Math.max(0, x - padding), Math.max(0, y - padding), width + padding * 2, height + padding * 2, Math.max(0, x - padding), Math.max(0, y - padding), width + padding * 2, height + padding * 2);
    }
    context.filter = 'none';
    const output = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.8 });
    bitmap.close();
    return await blobToBase64(output);
  } catch (error) {
    throw error instanceof Error ? error : new Error('SIH face-redaction failed');
  }
}
