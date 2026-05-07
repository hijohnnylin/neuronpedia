-- CreateTable
CREATE TABLE "NlaExplainCache" (
    "id" TEXT NOT NULL,
    "text" TEXT NOT NULL,
    "temperature" DOUBLE PRECISION NOT NULL,
    "modelId" TEXT NOT NULL,
    "nlaSourceName" TEXT NOT NULL,
    "resultJson" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "NlaExplainCache_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "NlaExplainCache_text_temperature_modelId_nlaSourceName_key" ON "NlaExplainCache"("text", "temperature", "modelId", "nlaSourceName");
