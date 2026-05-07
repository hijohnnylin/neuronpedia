/*
  Warnings:

  - You are about to drop the column `nlaSourceName` on the `NlaExplainCache` table. All the data in the column will be lost.
  - A unique constraint covering the columns `[text,numCompletionTokens,temperature,modelId,nlaSourceId,sortedPositions]` on the table `NlaExplainCache` will be added. If there are existing duplicate values, this will fail.
  - Added the required column `nlaSourceId` to the `NlaExplainCache` table without a default value. This is not possible if the table is not empty.

*/
-- DropIndex
DROP INDEX "NlaExplainCache_text_numCompletionTokens_temperature_modelI_key";

-- AlterTable
ALTER TABLE "NlaExplainCache" DROP COLUMN "nlaSourceName",
ADD COLUMN     "nlaSourceId" TEXT NOT NULL;

-- CreateIndex
CREATE INDEX "NlaExplainCache_nlaSourceId_idx" ON "NlaExplainCache"("nlaSourceId");

-- CreateIndex
CREATE UNIQUE INDEX "NlaExplainCache_text_numCompletionTokens_temperature_modelI_key" ON "NlaExplainCache"("text", "numCompletionTokens", "temperature", "modelId", "nlaSourceId", "sortedPositions");

-- AddForeignKey
ALTER TABLE "NlaExplainCache" ADD CONSTRAINT "NlaExplainCache_nlaSourceId_fkey" FOREIGN KEY ("nlaSourceId") REFERENCES "NlaSource"("id") ON DELETE CASCADE ON UPDATE CASCADE;
