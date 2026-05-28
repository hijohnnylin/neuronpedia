-- AlterTable
ALTER TABLE "ModelHeadMetrics" ADD COLUMN     "activationHistogram" JSONB,
ADD COLUMN     "headStatistics" JSONB,
ADD COLUMN     "qkDistanceHistogram" JSONB,
ADD COLUMN     "topKeyTokens" JSONB,
ADD COLUMN     "topQueryTokens" JSONB;

-- CreateTable
CREATE TABLE "ModelHeadSequence" (
    "id" TEXT NOT NULL,
    "modelId" TEXT NOT NULL,
    "layer" INTEGER NOT NULL,
    "headIndex" INTEGER NOT NULL,
    "modelName" TEXT NOT NULL,
    "datasetName" TEXT NOT NULL,
    "nSequences" INTEGER NOT NULL,
    "seqLen" INTEGER NOT NULL,
    "dtype" TEXT NOT NULL,
    "attnImplementation" TEXT NOT NULL,
    "interval" INTEGER NOT NULL,
    "tokens" TEXT[],
    "attentionIndices" INTEGER[],
    "attentionValues" DOUBLE PRECISION[],
    "maxActivation" DOUBLE PRECISION NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ModelHeadSequence_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "ModelHeadSequence_modelId_layer_headIndex_idx" ON "ModelHeadSequence"("modelId", "layer", "headIndex");

-- CreateIndex
CREATE INDEX "ModelHeadSequence_modelId_idx" ON "ModelHeadSequence"("modelId");

-- CreateIndex
CREATE INDEX "ModelHeadSequence_runConfigKey_idx" ON "ModelHeadSequence"("modelId", "datasetName", "nSequences", "seqLen", "dtype", "attnImplementation", "layer", "headIndex");

-- AddForeignKey
ALTER TABLE "ModelHeadSequence" ADD CONSTRAINT "ModelHeadSequence_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "Model"("id") ON DELETE CASCADE ON UPDATE CASCADE;
