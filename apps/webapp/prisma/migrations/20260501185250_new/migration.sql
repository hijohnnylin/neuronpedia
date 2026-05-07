/*
  Warnings:

  - The primary key for the `NlaSource` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - You are about to drop the column `name` on the `NlaSource` table. All the data in the column will be lost.

*/
-- DropForeignKey
ALTER TABLE "NlaExplainCache" DROP CONSTRAINT "NlaExplainCache_nlaSourceId_fkey";

-- DropIndex
DROP INDEX "NlaExplainCache_nlaSourceId_idx";

-- DropIndex
DROP INDEX "NlaSource_modelId_name_key";

-- AlterTable
ALTER TABLE "NlaSource" DROP CONSTRAINT "NlaSource_pkey",
DROP COLUMN "name",
ADD COLUMN     "displayName" TEXT NOT NULL DEFAULT '',
ADD CONSTRAINT "NlaSource_pkey" PRIMARY KEY ("modelId", "id");

-- CreateIndex
CREATE INDEX "NlaExplainCache_modelId_nlaSourceId_idx" ON "NlaExplainCache"("modelId", "nlaSourceId");

-- AddForeignKey
ALTER TABLE "NlaExplainCache" ADD CONSTRAINT "NlaExplainCache_modelId_nlaSourceId_fkey" FOREIGN KEY ("modelId", "nlaSourceId") REFERENCES "NlaSource"("modelId", "id") ON DELETE CASCADE ON UPDATE CASCADE;
