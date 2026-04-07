-- AlterTable
ALTER TABLE "Activation" ADD COLUMN     "zKIndices" INTEGER[] DEFAULT ARRAY[]::INTEGER[],
ADD COLUMN     "zQIndices" INTEGER[] DEFAULT ARRAY[]::INTEGER[],
ADD COLUMN     "zValues" DOUBLE PRECISION[] DEFAULT ARRAY[]::DOUBLE PRECISION[];

-- CreateTable
CREATE TABLE "NlaSource" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL DEFAULT '',
    "url" TEXT NOT NULL DEFAULT '',
    "author" TEXT NOT NULL DEFAULT '',
    "modelId" TEXT NOT NULL,
    "actor" TEXT NOT NULL,
    "critic" TEXT NOT NULL,
    "servers" TEXT[],
    "norm" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "NlaSource_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "NlaExplainCache" (
    "id" TEXT NOT NULL,
    "text" TEXT NOT NULL,
    "numCompletionTokens" INTEGER NOT NULL DEFAULT 0,
    "temperature" DOUBLE PRECISION NOT NULL,
    "modelId" TEXT NOT NULL,
    "nlaSourceName" TEXT NOT NULL,
    "resultJson" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "NlaExplainCache_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ActivationRaw" (
    "id" TEXT NOT NULL,
    "modelId" TEXT NOT NULL,
    "prompt" TEXT NOT NULL,
    "hookPoint" TEXT NOT NULL DEFAULT 'residual_stream',
    "captureType" TEXT NOT NULL DEFAULT 'final_output_token',
    "dtype" TEXT NOT NULL,
    "device" TEXT NOT NULL,
    "tokenStrings" TEXT[],
    "tokenIds" INTEGER[],
    "activations" JSONB NOT NULL,
    "creatorId" TEXT,
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ActivationRaw_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "NlaSource_modelId_idx" ON "NlaSource"("modelId");

-- CreateIndex
CREATE UNIQUE INDEX "NlaSource_modelId_name_key" ON "NlaSource"("modelId", "name");

-- CreateIndex
CREATE UNIQUE INDEX "NlaSource_modelId_actor_critic_key" ON "NlaSource"("modelId", "actor", "critic");

-- CreateIndex
CREATE UNIQUE INDEX "NlaExplainCache_text_numCompletionTokens_temperature_modelI_key" ON "NlaExplainCache"("text", "numCompletionTokens", "temperature", "modelId", "nlaSourceName");

-- CreateIndex
CREATE INDEX "ActivationRaw_modelId_idx" ON "ActivationRaw"("modelId");

-- CreateIndex
CREATE INDEX "ActivationRaw_creatorId_idx" ON "ActivationRaw"("creatorId");

-- CreateIndex
CREATE INDEX "ActivationRaw_prompt_idx" ON "ActivationRaw"("prompt");

-- AddForeignKey
ALTER TABLE "NlaSource" ADD CONSTRAINT "NlaSource_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "Model"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ActivationRaw" ADD CONSTRAINT "ActivationRaw_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "Model"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ActivationRaw" ADD CONSTRAINT "ActivationRaw_creatorId_fkey" FOREIGN KEY ("creatorId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;
