import { GetObjectCommand, ListObjectsV2Command, S3 } from '@aws-sdk/client-s3';
import fs from 'fs/promises';
import { ungzip } from 'node-gzip';
import { getLayerNumFromSource, getSourceSetNameFromSource } from './source';

export const DATASET_BUCKET = 'neuronpedia-datasets';
export const DATASET_REGION = 'us-east-1';
export const DATASET_VERSION = 'v1';
export const DATASET_BASE_PATH = `${DATASET_VERSION}/`;
export const CONFIG_DIR_NAME = 'config';
export const CONFIG_BASE_PATH = `${DATASET_BASE_PATH}${CONFIG_DIR_NAME}/`;
export const anonymousS3Client = new S3({
  region: DATASET_REGION,
  credentials: { accessKeyId: '', secretAccessKey: '' },
  signer: { sign: async (req) => req },
});

async function downloadFile(path: string, s3: boolean = true) {
  if (s3) {
    const response = await anonymousS3Client.send(
      new GetObjectCommand({
        Bucket: DATASET_BUCKET,
        Key: path,
      }),
    );
    return response.Body;
  }

  if (typeof window === 'undefined' && (path.startsWith('/') || path.startsWith('file:///'))) {
    try {
      const filePath = path.startsWith('file:///') ? path.slice(8) : path;
      const data = await fs.readFile(filePath, 'utf8');
      return {
        transformToString: () => data,
        transformToByteArray: () => Buffer.from(data),
      };
    } catch (error) {
      console.error('Error reading local file:', error);
      throw error;
    }
  }

  // Make fetch result compatible with the same interface
  return fetch(path).then((res) => {
    const text = res.text();
    return {
      transformToString: async () => text,
      transformToByteArray: async () => {
        const content = await text;
        return Buffer.from(content);
      },
    };
  });
}

export async function listBucketAtPath(path: string, delimiter?: string) {
  const response = await anonymousS3Client.send(
    new ListObjectsV2Command({
      Bucket: DATASET_BUCKET,
      Prefix: path,
      Delimiter: delimiter,
    }),
  );
  return response;
}

export function parseLine(line: string) {
  const parsedData = JSON.parse(line);
  if (parsedData.createdAt) {
    parsedData.createdAt = new Date(parsedData.createdAt);
  }
  if (parsedData.updatedAt) {
    parsedData.updatedAt = new Date(parsedData.updatedAt);
  }
  return parsedData;
}

export async function getS3ModelsToSources() {
  const s3ModelsResponse = await listBucketAtPath(DATASET_BASE_PATH, '/');
  const s3models = s3ModelsResponse.CommonPrefixes?.map((content) => content.Prefix || '') || [];

  const s3modelToSources: Record<string, string[]> = Object.fromEntries(
    await Promise.all(
      s3models.map(async (model) => {
        const response = await listBucketAtPath(model, '/');
        return [
          model.replace(DATASET_BASE_PATH, '').replace('/', ''),
          response.CommonPrefixes?.map((content) => content.Prefix?.split('/')[2]) || [],
        ];
      }),
    ),
  );
  Object.keys(s3modelToSources).forEach((model) => {
    if (model === CONFIG_DIR_NAME) {
      delete s3modelToSources[model];
    }
  });
  // Sort sources for each model
  Object.keys(s3modelToSources).forEach((model) => {
    s3modelToSources[model].sort((a, b) => {
      const sourceSetA = getSourceSetNameFromSource(a || '');
      const sourceSetB = getSourceSetNameFromSource(b || '');
      if (sourceSetA !== sourceSetB) {
        return sourceSetA.localeCompare(sourceSetB);
      }
      return getLayerNumFromSource(a || '') - getLayerNumFromSource(b || '');
    });
  });
  return s3modelToSources;
}

export async function getFilesInPath(path: string, filterByFileSuffix?: string): Promise<string[]> {
  const response = await anonymousS3Client.send(
    new ListObjectsV2Command({
      Bucket: DATASET_BUCKET,
      Prefix: path,
    }),
  );
  const files =
    response.Contents?.map((content) => content.Key || '')
      .filter(Boolean)
      .filter((file) => (filterByFileSuffix ? file.endsWith(filterByFileSuffix) : true)) || [];
  console.log('Files in path', files);
  return files;
}

export async function downloadAndDecompressFile(path: string, s3: boolean = true): Promise<string> {
  console.log('Downloading and decompressing file', path);
  const file = await downloadFile(path, s3);
  const arrayBuffer = await file?.transformToByteArray();
  return arrayBuffer ? (await ungzip(Buffer.from(arrayBuffer))).toString() : '';
}

async function downloadFileAsString(path: string): Promise<string> {
  const file = await downloadFile(path);
  if (!file) {
    throw new Error('File not found');
  }
  return file?.transformToString() || '';
}

export async function downloadFileJsonlParsedLines(path: string): Promise<any[]> {
  const file = await downloadFileAsString(path);
  return file.split('\n').filter(Boolean).map(parseLine);
}
