-- CreateTable
CREATE TABLE "NeuronTokenList" (
    "id" TEXT NOT NULL,
    "modelId" TEXT NOT NULL,
    "layer" TEXT NOT NULL,
    "index" TEXT NOT NULL,
    "typeName" TEXT NOT NULL,
    "creatorId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "pole" "ExplanationPole" NOT NULL,
    "aspect" "ExplanationAspect" NOT NULL,
    "tokens" TEXT[],
    "values" REAL[],
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "NeuronTokenList_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "NeuronTokenList_typeName_idx" ON "NeuronTokenList"("typeName");

-- CreateIndex
CREATE INDEX "NeuronTokenList_creatorId_idx" ON "NeuronTokenList"("creatorId");

-- CreateIndex
CREATE UNIQUE INDEX "NeuronTokenList_modelId_layer_index_typeName_name_pole_key" ON "NeuronTokenList"("modelId", "layer", "index", "typeName", "name", "pole");

-- AddForeignKey
ALTER TABLE "NeuronTokenList" ADD CONSTRAINT "NeuronTokenList_modelId_layer_index_fkey" FOREIGN KEY ("modelId", "layer", "index") REFERENCES "Neuron"("modelId", "layer", "index") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "NeuronTokenList" ADD CONSTRAINT "NeuronTokenList_typeName_fkey" FOREIGN KEY ("typeName") REFERENCES "ExplanationType"("name") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "NeuronTokenList" ADD CONSTRAINT "NeuronTokenList_creatorId_fkey" FOREIGN KEY ("creatorId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

