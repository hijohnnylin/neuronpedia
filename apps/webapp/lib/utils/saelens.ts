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

  for (let i = 0; i < bf16.length; i++) {
    view.setUint32(0, bf16[i] << 16, true);
    result[i] = view.getFloat32(0, true);
  }
  return result;
}

function convertF16ToF32(f16: Uint16Array): number[] {
  const result = new Array<number>(f16.length);

  for (let i = 0; i < f16.length; i++) {
    const h = f16[i];

    const sign = (h >> 15) & 0x1;

    const exponent = (h >> 10) & 0x1f;

    const mantissa = h & 0x3ff;
    if (exponent === 0) {
      result[i] = (sign ? -1 : 1) * 2 ** -14 * (mantissa / 1024);
    } else if (exponent === 31) {
      result[i] = mantissa === 0 ? (sign ? -Infinity : Infinity) : NaN;
    } else {
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
 * On-HuggingFace SAE checkpoint formats we know how to fetch a decoder row
 * from. The two formats agree on the actual W_dec layout (rows = latents,
 * cols = d_model) but disagree on the safetensors filename and the tensor
 * key used inside it. Adding a new format = one entry in `SAE_FORMAT_CONFIG`
 * below.
 */
export enum SAEFormat {
  // SAELens-native checkpoints (sae_weights.safetensors, uppercase W_dec).
  // This is the historical default and what every existing call site uses.
  SAELens = 'SAELens',
  // Google's gemma-scope-2 release. Despite the "gemma-scope-2" repo name,
  // SAELens loads these via its `gemma_3` loader. Layout:
  //   - file:   {folder}/params.safetensors
  //   - tensor: lowercase `w_dec`, shape (num_latents, d_model)
  // See sae_lens/loading/pretrained_sae_loaders.py:gemma_3_sae_huggingface_loader.
  GemmaScope2 = 'GemmaScope2',
}

interface SAEFormatConfig {
  /** Filename inside the HF folder that holds the SAE weights. */
  weightsFilename: string;
  /** Top-level safetensors tensor key for the decoder matrix. */
  decoderTensorKey: string;
}

const SAE_FORMAT_CONFIG: Record<SAEFormat, SAEFormatConfig> = {
  [SAEFormat.SAELens]: {
    weightsFilename: 'sae_weights.safetensors',
    decoderTensorKey: 'W_dec',
  },
  [SAEFormat.GemmaScope2]: {
    weightsFilename: 'params.safetensors',
    decoderTensorKey: 'w_dec',
  },
};

/**
 * Look up the on-disk layout for a given SAE checkpoint format. Useful when
 * the caller needs to construct the safetensors path themselves (e.g. the
 * `/api/nla/explain-saelens` route, which builds `${folder}/${weightsFilename}`
 * from a folder id stored in Postgres).
 */
export function getSAEFormatConfig(format: SAEFormat): SAEFormatConfig {
  return SAE_FORMAT_CONFIG[format];
}

/**
 * Fetches a single decoder latent vector (one row of the decoder matrix)
 * from an SAE safetensors file on HuggingFace, using HTTP Range requests
 * so only the header + one row of data are ever downloaded.
 *
 * @param format Optional. Picks which tensor key inside the safetensors
 *   header to read (SAELens → `W_dec`, GemmaScope2 → `w_dec`). Defaults to
 *   `SAEFormat.SAELens` to preserve the historical behavior of every
 *   existing caller. Note that `format` does NOT influence `path` —
 *   callers are still responsible for passing the correct safetensors
 *   filename (see `getSAEFormatConfig` for that).
 */
export async function fetchDecoderLatent(
  repo: string,
  path: string,
  index: number,
  format: SAEFormat = SAEFormat.SAELens,
): Promise<DecoderLatentResult> {
  if (!path.endsWith('.safetensors')) {
    throw new Error('path must point to a .safetensors file');
  }
  if (!Number.isInteger(index) || index < 0) {
    throw new Error('index must be a non-negative integer');
  }

  const { decoderTensorKey } = getSAEFormatConfig(format);

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

  const wDec = headerJson[decoderTensorKey];
  if (!wDec) {
    throw new Error(`Decoder tensor "${decoderTensorKey}" not found in this safetensors file (format=${format})`);
  }

  const { dtype, shape, data_offsets: dataOffsets } = wDec;
  const bytesPerElement = DTYPE_BYTES[dtype];
  if (!bytesPerElement) {
    throw new Error(`Unsupported dtype: ${dtype}`);
  }
  if (shape.length !== 2) {
    throw new Error(`Expected 2D ${decoderTensorKey} tensor, got ${shape.length}D`);
  }

  const [numLatents, dModel] = shape;
  if (index >= numLatents) {
    throw new Error(`index ${index} out of range [0, ${numLatents - 1}]`);
  }

  // Compute the exact byte range for one row of the decoder matrix
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
