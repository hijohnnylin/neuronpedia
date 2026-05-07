-- AlterTable
ALTER TABLE "NlaExplainCache" ADD COLUMN     "tokens" TEXT[] DEFAULT ARRAY[]::TEXT[];
