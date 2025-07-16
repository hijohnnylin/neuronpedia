/*
  Warnings:

  - Made the column `userId` on table `GraphMetadataSubgraph` required. This step will fail if there are existing NULL values in that column.

*/
-- AlterTable
ALTER TABLE "GraphMetadataSubgraph" ADD COLUMN     "displayName" TEXT,
ALTER COLUMN "userId" SET NOT NULL;
