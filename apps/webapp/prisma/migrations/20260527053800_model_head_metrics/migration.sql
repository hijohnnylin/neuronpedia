-- CreateTable
CREATE TABLE "ModelHeadMetrics" (
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
    "selfAttentionScore" DOUBLE PRECISION,
    "prevTokenScore" DOUBLE PRECISION,
    "patternEntropy" DOUBLE PRECISION,
    "qkDistance" DOUBLE PRECISION,
    "qkDistanceVariance" DOUBLE PRECISION,
    "inductionScore" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ModelHeadMetrics_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "ModelHeadMetrics_modelId_idx" ON "ModelHeadMetrics"("modelId");

-- CreateIndex
CREATE UNIQUE INDEX "ModelHeadMetrics_modelId_datasetName_nSequences_seqLen_dtype_att_key" ON "ModelHeadMetrics"("modelId", "datasetName", "nSequences", "seqLen", "dtype", "attnImplementation", "layer", "headIndex");

-- AddForeignKey
ALTER TABLE "ModelHeadMetrics" ADD CONSTRAINT "ModelHeadMetrics_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "Model"("id") ON DELETE CASCADE ON UPDATE CASCADE;
