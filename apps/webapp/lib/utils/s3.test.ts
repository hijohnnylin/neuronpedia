import { ListObjectsV2Command } from '@aws-sdk/client-s3';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { anonymousS3Client, getS3ModelsToSources, getS3SourcePath } from './s3';

describe('getS3SourcePath', () => {
  it('puts neuron layers under neurons/', () => {
    expect(getS3SourcePath('qwen3-8b', '17')).toBe('v1/qwen3-8b/neurons/17');
  });

  it('keeps other sources at the model root', () => {
    expect(getS3SourcePath('gpt2-small', '0-res-jb')).toBe('v1/gpt2-small/0-res-jb');
  });
});

describe('getS3ModelsToSources', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('lists the layers in neurons/ as bare layer sources', async () => {
    const folders: Record<string, string[]> = {
      'v1/': ['config', 'qwen3-8b'],
      'v1/qwen3-8b/': ['18-resid-batchtopk-65k__l0-80', 'neurons'],
      'v1/qwen3-8b/neurons/': ['10', '2'],
    };
    const list = async (command: ListObjectsV2Command) => {
      const prefix = command.input.Prefix || '';
      return { CommonPrefixes: (folders[prefix] || []).map((name) => ({ Prefix: `${prefix}${name}/` })) };
    };
    // `send` is overloaded, so the fake needs a cast.
    vi.spyOn(anonymousS3Client, 'send').mockImplementation(list as unknown as typeof anonymousS3Client.send);

    expect(await getS3ModelsToSources()).toEqual({
      'qwen3-8b': ['2', '10', '18-resid-batchtopk-65k__l0-80'],
    });
  });
});
