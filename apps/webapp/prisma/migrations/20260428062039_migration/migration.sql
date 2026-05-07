/*
  Warnings:

  - You are about to drop the column `actor` on the `NlaSource` table. All the data in the column will be lost.
  - You are about to drop the column `critic` on the `NlaSource` table. All the data in the column will be lost.
  - You are about to drop the `ActivationRaw` table. If the table is not empty, all the data it contains will be lost.
  - A unique constraint covering the columns `[text,numCompletionTokens,temperature,modelId,nlaSourceName,sortedPositions]` on the table `NlaExplainCache` will be added. If there are existing duplicate values, this will fail.
  - A unique constraint covering the columns `[modelId,ar,av,layerNum]` on the table `NlaSource` will be added. If there are existing duplicate values, this will fail.
  - Added the required column `ar` to the `NlaSource` table without a default value. This is not possible if the table is not empty.
  - Added the required column `av` to the `NlaSource` table without a default value. This is not possible if the table is not empty.
  - Added the required column `layerNum` to the `NlaSource` table without a default value. This is not possible if the table is not empty.

*/
-- CreateEnum
CREATE TYPE "ProbeSampleType" AS ENUM ('DECEPTION');

-- CreateEnum
CREATE TYPE "ProbeArchitecture" AS ENUM ('LINEAR', 'MLP');

-- CreateEnum
CREATE TYPE "TokenSelection" AS ENUM ('LAST_ASSISTANT', 'ALL_TOKENS', 'EOT_TOKEN', 'REASONING_ONLY');

-- CreateEnum
CREATE TYPE "ReductionMethod" AS ENUM ('MEAN_POOL', 'LAST_TOKEN', 'MAX_POOL');

-- CreateEnum
CREATE TYPE "ProbeLogAction" AS ENUM ('NEW_PROBE', 'NEW_SAMPLE', 'SAMPLE_ACCEPTED', 'SAMPLE_REJECTED', 'RANK_CHANGE');

-- DropForeignKey
ALTER TABLE "ActivationRaw" DROP CONSTRAINT "ActivationRaw_creatorId_fkey";

-- DropForeignKey
ALTER TABLE "ActivationRaw" DROP CONSTRAINT "ActivationRaw_modelId_fkey";

-- DropIndex
DROP INDEX "NlaExplainCache_text_numCompletionTokens_temperature_modelI_key";

-- DropIndex
DROP INDEX "NlaSource_modelId_actor_critic_key";

-- AlterTable
ALTER TABLE "NlaExplainCache" ADD COLUMN     "sortedPositions" INTEGER[];

-- AlterTable
ALTER TABLE "NlaSource" DROP COLUMN "actor",
DROP COLUMN "critic",
ADD COLUMN     "ar" TEXT NOT NULL,
ADD COLUMN     "av" TEXT NOT NULL,
ADD COLUMN     "layerNum" INTEGER NOT NULL;

-- DropTable
DROP TABLE "ActivationRaw";

-- CreateTable
CREATE TABLE "SampleDataset" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "url" TEXT,
    "createdByUserId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "SampleDataset_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Sample" (
    "id" TEXT NOT NULL,
    "type" "ProbeSampleType" NOT NULL,
    "modelId" TEXT NOT NULL,
    "datasetId" TEXT NOT NULL,
    "sampleIndex" INTEGER NOT NULL,
    "groundTruth" BOOLEAN NOT NULL,
    "conversation" JSONB NOT NULL,
    "conversationTokenized" INTEGER[],
    "conversationTokenizedStr" TEXT[],
    "finalResponseStartIndex" INTEGER NOT NULL,
    "conversationEmbedding" vector(256),
    "userId" TEXT NOT NULL,
    "elo" DOUBLE PRECISION NOT NULL DEFAULT 1500,
    "vettedAt" TIMESTAMP(3),
    "vettedBy" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Sample_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Probe" (
    "id" TEXT NOT NULL,
    "customLabel" TEXT NOT NULL,
    "modelId" TEXT NOT NULL,
    "type" "ProbeSampleType" NOT NULL,
    "userId" TEXT NOT NULL,
    "architecture" "ProbeArchitecture" NOT NULL DEFAULT 'LINEAR',
    "tokenSelection" "TokenSelection" NOT NULL DEFAULT 'LAST_ASSISTANT',
    "reductionMethod" "ReductionMethod" NOT NULL DEFAULT 'MEAN_POOL',
    "layers" INTEGER[],
    "weights" DOUBLE PRECISION[],
    "bias" DOUBLE PRECISION NOT NULL,
    "weightsBlob" BYTEA,
    "elo" DOUBLE PRECISION NOT NULL DEFAULT 1500,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "creatorNote" TEXT,

    CONSTRAINT "Probe_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ProbeMatch" (
    "id" TEXT NOT NULL,
    "probeId" TEXT NOT NULL,
    "sampleId" TEXT NOT NULL,
    "probeCorrect" BOOLEAN NOT NULL,
    "probeScore" DOUBLE PRECISION NOT NULL,
    "tokenScores" DOUBLE PRECISION[],
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ProbeMatch_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ProbeLog" (
    "id" TEXT NOT NULL,
    "action" "ProbeLogAction" NOT NULL,
    "modelId" TEXT NOT NULL,
    "userId" TEXT,
    "probeId" TEXT,
    "sampleId" TEXT,
    "details" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ProbeLog_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ProcessLock" (
    "name" TEXT NOT NULL,
    "startedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expiresAt" TIMESTAMP(3) NOT NULL,
    "details" JSONB,

    CONSTRAINT "ProcessLock_pkey" PRIMARY KEY ("name")
);

-- CreateIndex
CREATE UNIQUE INDEX "SampleDataset_name_key" ON "SampleDataset"("name");

-- CreateIndex
CREATE INDEX "SampleDataset_createdByUserId_idx" ON "SampleDataset"("createdByUserId");

-- CreateIndex
CREATE INDEX "Sample_type_idx" ON "Sample"("type");

-- CreateIndex
CREATE INDEX "Sample_modelId_idx" ON "Sample"("modelId");

-- CreateIndex
CREATE INDEX "Sample_datasetId_idx" ON "Sample"("datasetId");

-- CreateIndex
CREATE INDEX "Sample_userId_idx" ON "Sample"("userId");

-- CreateIndex
CREATE INDEX "Sample_elo_idx" ON "Sample"("elo");

-- CreateIndex
CREATE UNIQUE INDEX "Sample_modelId_datasetId_sampleIndex_key" ON "Sample"("modelId", "datasetId", "sampleIndex");

-- CreateIndex
CREATE UNIQUE INDEX "Probe_customLabel_key" ON "Probe"("customLabel");

-- CreateIndex
CREATE INDEX "Probe_type_idx" ON "Probe"("type");

-- CreateIndex
CREATE INDEX "Probe_modelId_idx" ON "Probe"("modelId");

-- CreateIndex
CREATE INDEX "Probe_userId_idx" ON "Probe"("userId");

-- CreateIndex
CREATE INDEX "Probe_elo_idx" ON "Probe"("elo");

-- CreateIndex
CREATE INDEX "ProbeMatch_probeId_idx" ON "ProbeMatch"("probeId");

-- CreateIndex
CREATE INDEX "ProbeMatch_sampleId_probeCorrect_idx" ON "ProbeMatch"("sampleId", "probeCorrect");

-- CreateIndex
CREATE INDEX "ProbeLog_modelId_idx" ON "ProbeLog"("modelId");

-- CreateIndex
CREATE INDEX "ProbeLog_action_idx" ON "ProbeLog"("action");

-- CreateIndex
CREATE INDEX "ProbeLog_createdAt_idx" ON "ProbeLog"("createdAt");

-- CreateIndex
CREATE UNIQUE INDEX "NlaExplainCache_text_numCompletionTokens_temperature_modelI_key" ON "NlaExplainCache"("text", "numCompletionTokens", "temperature", "modelId", "nlaSourceName", "sortedPositions");

-- CreateIndex
CREATE UNIQUE INDEX "NlaSource_modelId_ar_av_layerNum_key" ON "NlaSource"("modelId", "ar", "av", "layerNum");

-- AddForeignKey
ALTER TABLE "Sample" ADD CONSTRAINT "Sample_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "Model"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Sample" ADD CONSTRAINT "Sample_datasetId_fkey" FOREIGN KEY ("datasetId") REFERENCES "SampleDataset"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Sample" ADD CONSTRAINT "Sample_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Probe" ADD CONSTRAINT "Probe_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "Model"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Probe" ADD CONSTRAINT "Probe_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProbeMatch" ADD CONSTRAINT "ProbeMatch_probeId_fkey" FOREIGN KEY ("probeId") REFERENCES "Probe"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProbeMatch" ADD CONSTRAINT "ProbeMatch_sampleId_fkey" FOREIGN KEY ("sampleId") REFERENCES "Sample"("id") ON DELETE CASCADE ON UPDATE CASCADE;
