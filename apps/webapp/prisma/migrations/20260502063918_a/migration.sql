/*
  Warnings:

  - A unique constraint covering the columns `[textHash,numCompletionTokens,temperature,modelId,nlaSourceId,sortedPositions]` on the table `NlaExplainCache` will be added. If there are existing duplicate values, this will fail.
  - Made the column `textHash` on table `NlaExplainCache` required. This step will fail if there are existing NULL values in that column.

*/
-- DropIndex
DROP INDEX "NlaExplainCache_text_numCompletionTokens_temperature_modelI_key";

-- AlterTable
ALTER TABLE "NlaExplainCache" ALTER COLUMN "textHash" SET NOT NULL,
ALTER COLUMN "textHash" SET DEFAULT '';

-- CreateIndex
CREATE UNIQUE INDEX "NlaExplainCache_textHash_numCompletionTokens_temperature_mo_key" ON "NlaExplainCache"("textHash", "numCompletionTokens", "temperature", "modelId", "nlaSourceId", "sortedPositions");
