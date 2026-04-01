/*
  Warnings:

  - You are about to drop the column `type` on the `ProblemNode` table. All the data in the column will be lost.
  - The `nodeTypes` column on the `ProblemNode` table would be dropped and recreated. This will lead to data loss if there is data in the column.

*/
-- DropIndex
DROP INDEX "ProblemNode_type_idx";

-- AlterTable
ALTER TABLE "ProblemNode" DROP COLUMN "type",
DROP COLUMN "nodeTypes",
ADD COLUMN     "nodeTypes" "ProblemNodeType"[] DEFAULT ARRAY['topic']::"ProblemNodeType"[];
