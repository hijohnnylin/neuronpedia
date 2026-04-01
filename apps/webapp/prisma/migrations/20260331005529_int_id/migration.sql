/*
  Warnings:

  - The primary key for the `ProblemNode` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - The `id` column on the `ProblemNode` table would be dropped and recreated. This will lead to data loss if there is data in the column.
  - The `parentId` column on the `ProblemNode` table would be dropped and recreated. This will lead to data loss if there is data in the column.
  - Changed the type of `sourceNodeId` on the `ProblemEdge` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.
  - Changed the type of `targetNodeId` on the `ProblemEdge` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.
  - Changed the type of `problemNodeId` on the `ProblemNodeComment` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.
  - Changed the type of `problemNodeId` on the `ProblemNodeLog` table. No cast exists, the column would be dropped and recreated, which cannot be done if there is data, since the column is required.

*/
-- DropForeignKey
ALTER TABLE "ProblemEdge" DROP CONSTRAINT "ProblemEdge_sourceNodeId_fkey";

-- DropForeignKey
ALTER TABLE "ProblemEdge" DROP CONSTRAINT "ProblemEdge_targetNodeId_fkey";

-- DropForeignKey
ALTER TABLE "ProblemNode" DROP CONSTRAINT "ProblemNode_parentId_fkey";

-- DropForeignKey
ALTER TABLE "ProblemNodeComment" DROP CONSTRAINT "ProblemNodeComment_problemNodeId_fkey";

-- DropForeignKey
ALTER TABLE "ProblemNodeLog" DROP CONSTRAINT "ProblemNodeLog_problemNodeId_fkey";

-- AlterTable
ALTER TABLE "ProblemEdge" DROP COLUMN "sourceNodeId",
ADD COLUMN     "sourceNodeId" INTEGER NOT NULL,
DROP COLUMN "targetNodeId",
ADD COLUMN     "targetNodeId" INTEGER NOT NULL;

-- AlterTable
ALTER TABLE "ProblemNode" DROP CONSTRAINT "ProblemNode_pkey",
DROP COLUMN "id",
ADD COLUMN     "id" SERIAL NOT NULL,
DROP COLUMN "parentId",
ADD COLUMN     "parentId" INTEGER,
ADD CONSTRAINT "ProblemNode_pkey" PRIMARY KEY ("id");

-- AlterTable
ALTER TABLE "ProblemNodeComment" DROP COLUMN "problemNodeId",
ADD COLUMN     "problemNodeId" INTEGER NOT NULL;

-- AlterTable
ALTER TABLE "ProblemNodeLog" DROP COLUMN "problemNodeId",
ADD COLUMN     "problemNodeId" INTEGER NOT NULL;

-- CreateIndex
CREATE INDEX "ProblemEdge_sourceNodeId_idx" ON "ProblemEdge"("sourceNodeId");

-- CreateIndex
CREATE INDEX "ProblemEdge_targetNodeId_idx" ON "ProblemEdge"("targetNodeId");

-- CreateIndex
CREATE UNIQUE INDEX "ProblemEdge_sourceNodeId_targetNodeId_type_key" ON "ProblemEdge"("sourceNodeId", "targetNodeId", "type");

-- CreateIndex
CREATE INDEX "ProblemNode_parentId_idx" ON "ProblemNode"("parentId");

-- CreateIndex
CREATE INDEX "ProblemNodeComment_problemNodeId_idx" ON "ProblemNodeComment"("problemNodeId");

-- CreateIndex
CREATE INDEX "ProblemNodeLog_problemNodeId_idx" ON "ProblemNodeLog"("problemNodeId");

-- AddForeignKey
ALTER TABLE "ProblemNode" ADD CONSTRAINT "ProblemNode_parentId_fkey" FOREIGN KEY ("parentId") REFERENCES "ProblemNode"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProblemEdge" ADD CONSTRAINT "ProblemEdge_sourceNodeId_fkey" FOREIGN KEY ("sourceNodeId") REFERENCES "ProblemNode"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProblemEdge" ADD CONSTRAINT "ProblemEdge_targetNodeId_fkey" FOREIGN KEY ("targetNodeId") REFERENCES "ProblemNode"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProblemNodeComment" ADD CONSTRAINT "ProblemNodeComment_problemNodeId_fkey" FOREIGN KEY ("problemNodeId") REFERENCES "ProblemNode"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProblemNodeLog" ADD CONSTRAINT "ProblemNodeLog_problemNodeId_fkey" FOREIGN KEY ("problemNodeId") REFERENCES "ProblemNode"("id") ON DELETE CASCADE ON UPDATE CASCADE;
