/*
  Warnings:

  - Made the column `userId` on table `GraphMetadataDataPutRequest` required. This step will fail if there are existing NULL values in that column.

*/
-- AlterTable
ALTER TABLE "GraphMetadataDataPutRequest" ALTER COLUMN "userId" SET NOT NULL;
