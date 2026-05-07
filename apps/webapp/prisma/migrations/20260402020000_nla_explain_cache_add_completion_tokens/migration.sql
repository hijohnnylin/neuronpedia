-- AlterTable
ALTER TABLE "NlaExplainCache" ADD COLUMN "numCompletionTokens" INTEGER NOT NULL DEFAULT 0;

-- DropIndex
DROP INDEX "NlaExplainCache_text_temperature_modelId_nlaSourceName_key";

-- CreateIndex
CREATE UNIQUE INDEX "NlaExplainCache_text_numCompletionTokens_temperature_modelI_key" ON "NlaExplainCache"("text", "numCompletionTokens", "temperature", "modelId", "nlaSourceName");
