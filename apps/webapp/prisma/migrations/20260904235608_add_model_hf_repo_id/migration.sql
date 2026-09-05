/*
  Warnings:

  - A unique constraint covering the columns `[hfRepoId]` on the table `Model` will be added. If there are existing duplicate values, this will fail.

*/
-- AlterTable
ALTER TABLE "Model" ADD COLUMN     "hfRepoId" TEXT;

UPDATE "Model" SET "hfRepoId" = 'EleutherAI/pythia-70m-deduped' WHERE id = 'pythia-70m-deduped';
UPDATE "Model" SET "hfRepoId" = 'openai-community/gpt2' WHERE id = 'gpt2-small';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-3-270m' WHERE id = 'gemma-3-270m';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-3-270m-it' WHERE id = 'gemma-3-270m-it';
UPDATE "Model" SET "hfRepoId" = 'Qwen/Qwen3.5-0.8B' WHERE id = 'qwen3.5-0.8b';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-3-1b-pt' WHERE id = 'gemma-3-1b';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-3-1b-it' WHERE id = 'gemma-3-1b-it';
UPDATE "Model" SET "hfRepoId" = 'Qwen/Qwen2.5-1.5B-Instruct' WHERE id = 'qwen2.5-1.5b-it';
UPDATE "Model" SET "hfRepoId" = 'Qwen/Qwen3-1.7B' WHERE id = 'qwen3-1.7b';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-2-2b' WHERE id = 'gemma-2-2b';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-2-2b-it' WHERE id = 'gemma-2-2b-it';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-4-E2B' WHERE id = 'gemma-4-e2b';
UPDATE "Model" SET "hfRepoId" = 'Qwen/Qwen3.5-2B-Base' WHERE id = 'qwen3.5-2b-pt';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-3-4b-pt' WHERE id = 'gemma-3-4b';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-3-4b-it' WHERE id = 'gemma-3-4b-it';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-4-E4B' WHERE id = 'gemma-4-e4b';
UPDATE "Model" SET "hfRepoId" = 'Qwen/Qwen3-4B' WHERE id = 'qwen3-4b';
UPDATE "Model" SET "hfRepoId" = 'Qwen/Qwen3.5-4B' WHERE id = 'qwen3.5-4b';
UPDATE "Model" SET "hfRepoId" = 'allenai/Olmo-3-1025-7B' WHERE id = 'olmo-3-1025-7b';
UPDATE "Model" SET "hfRepoId" = 'Qwen/Qwen2.5-7B-Instruct' WHERE id = 'qwen2.5-7b-it';
UPDATE "Model" SET "hfRepoId" = 'meta-llama/Llama-3.1-8B' WHERE id = 'llama3.1-8b';
UPDATE "Model" SET "hfRepoId" = 'meta-llama/Llama-3.1-8B-Instruct' WHERE id = 'llama3.1-8b-it';
UPDATE "Model" SET "hfRepoId" = 'Qwen/Qwen3-8B' WHERE id = 'qwen3-8b';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-2-9b' WHERE id = 'gemma-2-9b';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-2-9b-it' WHERE id = 'gemma-2-9b-it';
UPDATE "Model" SET "hfRepoId" = 'Qwen/Qwen3.5-9B-Base' WHERE id = 'qwen3.5-9b-pt';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-3-12b-pt' WHERE id = 'gemma-3-12b';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-3-12b-it' WHERE id = 'gemma-3-12b-it';
UPDATE "Model" SET "hfRepoId" = 'Qwen/Qwen3-14B' WHERE id = 'qwen3-14b';
UPDATE "Model" SET "hfRepoId" = 'openai/gpt-oss-20b' WHERE id = 'gpt-oss-20b';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-2-27b' WHERE id = 'gemma-2-27b';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-3-27b-pt' WHERE id = 'gemma-3-27b';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-3-27b-it' WHERE id = 'gemma-3-27b-it';
UPDATE "Model" SET "hfRepoId" = 'Qwen/Qwen3.5-27B' WHERE id = 'qwen3.5-27b';
UPDATE "Model" SET "hfRepoId" = 'Qwen/Qwen3.6-27B' WHERE id = 'qwen3.6-27b';
UPDATE "Model" SET "hfRepoId" = 'meta-models/Muse-Glimmer-30B' WHERE id = 'glimmer-30b';
UPDATE "Model" SET "hfRepoId" = 'google/gemma-4-31B' WHERE id = 'gemma-4-31b';
UPDATE "Model" SET "hfRepoId" = 'allenai/Olmo-3-1125-32B' WHERE id = 'olmo-3-1125-32b';
UPDATE "Model" SET "hfRepoId" = 'Qwen/Qwen3-32B' WHERE id = 'qwen3-32b';
UPDATE "Model" SET "hfRepoId" = 'meta-llama/Llama-3.3-70B-Instruct' WHERE id = 'llama3.3-70b-it';
UPDATE "Model" SET "hfRepoId" = 'deepseek-ai/DeepSeek-V4-Flash' WHERE id = 'deepseek-v4-flash';
UPDATE "Model" SET "hfRepoId" = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' WHERE id = 'deepseek-r1-distill-llama-8b';

-- CreateIndex
CREATE UNIQUE INDEX "Model_hfRepoId_key" ON "Model"("hfRepoId");
