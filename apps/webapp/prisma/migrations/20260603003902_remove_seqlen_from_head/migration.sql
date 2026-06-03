/*
  Warnings:

  - You are about to drop the column `seqLen` on the `ModelHeadSequence` table. All the data in the column will be lost.

*/
-- DropIndex
DROP INDEX "ModelHeadSequence_runConfigKey_idx";

-- AlterTable
ALTER TABLE "ModelHeadSequence" DROP COLUMN "seqLen";

-- CreateIndex
CREATE INDEX "ModelHeadSequence_runConfigKey_idx" ON "ModelHeadSequence"("modelId", "datasetName", "nSequences", "dtype", "attnImplementation");
