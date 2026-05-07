import { HF_TOKEN } from '@/lib/env';

const DTYPE_BYTES: Record<string, number> = {
  F32: 4,
  F16: 2,
  BF16: 2,
  F64: 8,
};

function convertBF16ToF32(bf16: Uint16Array): number[] {
  const result = new Array<number>(bf16.length);
  const buf = new ArrayBuffer(4);
  const view = new DataView(buf);
  // eslint-disable-next-line no-plusplus
  for (let i = 0; i < bf16.length; i++) {
    // eslint-disable-next-line no-bitwise
    view.setUint32(0, bf16[i] << 16, true);
    result[i] = view.getFloat32(0, true);
  }
  return result;
}

function convertF16ToF32(f16: Uint16Array): number[] {
  const result = new Array<number>(f16.length);
  // eslint-disable-next-line no-plusplus
  for (let i = 0; i < f16.length; i++) {
    const h = f16[i];
    // eslint-disable-next-line no-bitwise
    const sign = (h >> 15) & 0x1;
    // eslint-disable-next-line no-bitwise
    const exponent = (h >> 10) & 0x1f;
    // eslint-disable-next-line no-bitwise
    const mantissa = h & 0x3ff;
    if (exponent === 0) {
      // eslint-disable-next-line no-restricted-properties
      result[i] = (sign ? -1 : 1) * 2 ** -14 * (mantissa / 1024);
    } else if (exponent === 31) {
      result[i] = mantissa === 0 ? (sign ? -Infinity : Infinity) : NaN;
    } else {
      // eslint-disable-next-line no-restricted-properties
      result[i] = (sign ? -1 : 1) * 2 ** (exponent - 15) * (1 + mantissa / 1024);
    }
  }
  return result;
}

function rawToFloat(buf: ArrayBuffer, dtype: string): number[] {
  if (dtype === 'F32') return Array.from(new Float32Array(buf));
  if (dtype === 'F16') return convertF16ToF32(new Uint16Array(buf));
  if (dtype === 'BF16') return convertBF16ToF32(new Uint16Array(buf));
  return Array.from(new Float64Array(buf));
}

async function fetchRange(url: string, start: number, end: number): Promise<Response> {
  const headers: HeadersInit = { Range: `bytes=${start}-${end}` };
  if (HF_TOKEN) {
    headers.Authorization = `Bearer ${HF_TOKEN}`;
  }
  return fetch(url, { headers });
}

export interface DecoderLatentResult {
  repo: string;
  path: string;
  index: number;
  dtype: string;
  num_latents: number;
  d_model: number;
  values: number[];
}

/**
 * Fetches a single decoder latent vector (one row of W_dec) from a SAELens
 * safetensors file on HuggingFace, using HTTP Range requests so only the
 * header + one row of data are ever downloaded.
 */
export async function fetchDecoderLatent(repo: string, path: string, index: number): Promise<DecoderLatentResult> {
  if (!path.endsWith('.safetensors')) {
    throw new Error('path must point to a .safetensors file');
  }
  if (!Number.isInteger(index) || index < 0) {
    throw new Error('index must be a non-negative integer');
  }

  const fileUrl = `https://huggingface.co/${repo}/resolve/main/${path}`;

  // Fetch the 8-byte header length prefix
  const sizeRes = await fetchRange(fileUrl, 0, 7);
  if (!sizeRes.ok) {
    throw new Error(`Failed to access safetensors file (HTTP ${sizeRes.status}). Check repo/path.`);
  }
  const sizeBuf = await sizeRes.arrayBuffer();
  const headerSize = Number(new DataView(sizeBuf).getBigUint64(0, true));

  // Fetch the JSON header describing all tensors
  const headerRes = await fetchRange(fileUrl, 8, 8 + headerSize - 1);
  if (!headerRes.ok) {
    throw new Error(`Failed to read safetensors header (HTTP ${headerRes.status})`);
  }
  const headerJson = JSON.parse(await headerRes.text());

  const wDec = headerJson.W_dec;
  if (!wDec) {
    throw new Error('W_dec tensor not found in this safetensors file');
  }

  const { dtype, shape, data_offsets: dataOffsets } = wDec;
  const bytesPerElement = DTYPE_BYTES[dtype];
  if (!bytesPerElement) {
    throw new Error(`Unsupported dtype: ${dtype}`);
  }
  if (shape.length !== 2) {
    throw new Error(`Expected 2D W_dec tensor, got ${shape.length}D`);
  }

  const [numLatents, dModel] = shape;
  if (index >= numLatents) {
    throw new Error(`index ${index} out of range [0, ${numLatents - 1}]`);
  }

  // Compute the exact byte range for one row of W_dec
  const dataStart = 8 + headerSize + dataOffsets[0];
  const rowBytes = dModel * bytesPerElement;
  const rowStart = dataStart + index * rowBytes;
  const rowEnd = rowStart + rowBytes - 1;

  const rowRes = await fetchRange(fileUrl, rowStart, rowEnd);
  if (!rowRes.ok) {
    throw new Error(`Failed to fetch latent vector (HTTP ${rowRes.status})`);
  }
  const rowBuf = await rowRes.arrayBuffer();

  return {
    repo,
    path,
    index,
    dtype,
    num_latents: numLatents,
    d_model: dModel,
    values: rawToFloat(rowBuf, dtype),
  };
}
